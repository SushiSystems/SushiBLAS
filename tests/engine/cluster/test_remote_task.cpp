/**************************************************************************/
/* test_remote_task.cpp                                                   */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file test_remote_task.cpp
 * @brief Tests that a pack of every type round-trips through a worker and a stale worker refuses.
 * @author Mustafa Garip
 */

#include <atomic>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/cluster/bound_pack.hpp>
#include <SushiBLAS/cluster/cluster_binding.hpp>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiBLAS/cluster/remote_task.hpp>
#include <SushiBLAS/cluster/tensor_pack.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiRuntime/distributed/distributed.hpp>
#include <SushiRuntime/distributed/in_process_transport.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using SushiRuntime::Distributed::BufferView;
            using SushiRuntime::Distributed::Cluster;
            using SushiRuntime::Distributed::ClusterOptions;
            using SushiRuntime::Distributed::InProcessFabric;
            using SushiRuntime::Distributed::InProcessTransport;
            using SushiRuntime::Distributed::KernelRegistry;
            using SushiRuntime::Distributed::WorkerAgent;

            /** @brief Holds how long a test waits for a worker to join. */
            constexpr std::chrono::milliseconds K_WAIT{10000};

            /** @brief Holds the kernel's operation name, before its fingerprint. */
            constexpr const char* K_OPERATION_NAME = "sushiblas.test.pack_transform";

            /** @brief Holds the elements of each test tensor. */
            constexpr std::int64_t K_ELEMENTS = 37;

            /** @brief Returns the known transform of one scalar: doubled, or 3x + 1 wrapping. */
            template<typename Scalar>
            Scalar transformed(Scalar value)
            {
                if constexpr (std::is_integral_v<Scalar>)
                {
                    using Unsigned = std::make_unsigned_t<Scalar>;
                    return static_cast<Scalar>(static_cast<Unsigned>(value) * Unsigned{3} +
                                               Unsigned{1});
                }
                else
                {
                    return static_cast<Scalar>(value + value);
                }
            }

            /** @brief Submits the transform of count scalars after previous. */
            template<typename Scalar>
            sycl::event submit_transform(sycl::queue& queue, Scalar* values, std::size_t count,
                                         const std::vector<sycl::event>& previous)
            {
                return queue.submit([&](sycl::handler& handler)
                {
                    handler.depends_on(previous);
                    handler.parallel_for(sycl::range<1>(count), [=](sycl::id<1> index)
                    {
                        values[index[0]] = transformed(values[index[0]]);
                    });
                });
            }

            /** @brief Submits the transform of one segment; a complex segment is its scalars. */
            sycl::event submit_segment(sycl::queue& queue, const PackSpan& span,
                                       const std::vector<sycl::event>& previous)
            {
                switch (span.dtype)
                {
                    case Core::DataType::Half:
                        return submit_transform(queue, span.as<sycl::half>(), span.count,
                                                previous);
                    case Core::DataType::Float32:
                        return submit_transform(queue, span.as<float>(), span.count, previous);
                    case Core::DataType::Float64:
                        return submit_transform(queue, span.as<double>(), span.count, previous);
                    case Core::DataType::Complex32:
                        return submit_transform(
                            queue, reinterpret_cast<float*>(span.as<std::complex<float>>()),
                            span.count * 2, previous);
                    case Core::DataType::Complex64:
                        return submit_transform(
                            queue, reinterpret_cast<double*>(span.as<std::complex<double>>()),
                            span.count * 2, previous);
                    case Core::DataType::Int32:
                        return submit_transform(queue, span.as<std::int32_t>(), span.count,
                                                previous);
                    case Core::DataType::Int64:
                        return submit_transform(queue, span.as<std::int64_t>(), span.count,
                                                previous);
                }
                return sycl::event();
            }

            /** @brief Returns one tensor of K_ELEMENTS elements per data type. */
            std::vector<Tensor> one_tensor_per_type(SushiRuntime::Runtime& runtime)
            {
                std::vector<Tensor> tensors;
                for (std::size_t type = 0; type < K_PACK_SEGMENT_COUNT; ++type)
                {
                    const Core::DataType dtype = static_cast<Core::DataType>(type);
                    const std::int64_t dims[] = {K_ELEMENTS};
                    const std::size_t bytes =
                        static_cast<std::size_t>(K_ELEMENTS) * Core::bytes_per_element(dtype);
                    tensors.emplace_back(
                        SushiRuntime::make_sushi<Storage>(runtime, bytes,
                                                          SushiRuntime::Residency::Shared, 0),
                        SushiRuntime::span<const std::int64_t>(dims, 1), dtype);
                }
                return tensors;
            }

            /** @brief Writes into each tensor scalars no transform leaves unchanged. */
            void fill(std::vector<Tensor>& tensors)
            {
                for (Tensor& tensor : tensors)
                {
                    auto* bytes = static_cast<std::uint8_t*>(tensor.data());
                    const std::size_t size = static_cast<std::size_t>(tensor.num_elements) *
                                             Core::bytes_per_element(tensor.dtype);
                    const bool is_integer = tensor.dtype == Core::DataType::Int32 ||
                                            tensor.dtype == Core::DataType::Int64;
                    if (is_integer)
                    {
                        for (std::size_t index = 0; index < size; ++index)
                            bytes[index] = static_cast<std::uint8_t>(index * 37 + 11);
                        continue;
                    }
                    const std::size_t scalars = size / Core::bytes_per_element(tensor.dtype) *
                                                (tensor.dtype == Core::DataType::Complex32 ||
                                                         tensor.dtype == Core::DataType::Complex64
                                                     ? 2
                                                     : 1);
                    for (std::size_t index = 0; index < scalars; ++index)
                    {
                        const float value = 0.25f + static_cast<float>(index % 13);
                        switch (tensor.dtype)
                        {
                            case Core::DataType::Half:
                                tensor.data_as<sycl::half>()[index] = sycl::half(value);
                                break;
                            case Core::DataType::Float32:
                            case Core::DataType::Complex32:
                                tensor.data_as<float>()[index] = value;
                                break;
                            default:
                                tensor.data_as<double>()[index] = value;
                                break;
                        }
                    }
                }
            }

            /** @brief Returns the bytes of a tensor after the transform, computed on the host. */
            std::vector<std::uint8_t> expected_bytes(const Tensor& tensor)
            {
                const std::size_t count = static_cast<std::size_t>(tensor.num_elements);
                const std::size_t size = count * Core::bytes_per_element(tensor.dtype);
                std::vector<std::uint8_t> bytes(size);
                std::memcpy(bytes.data(), tensor.data(), size);
                auto apply = [](auto* scalars, std::size_t scalar_count)
                {
                    for (std::size_t index = 0; index < scalar_count; ++index)
                        scalars[index] = transformed(scalars[index]);
                };
                switch (tensor.dtype)
                {
                    case Core::DataType::Half:
                        apply(reinterpret_cast<sycl::half*>(bytes.data()), count);
                        break;
                    case Core::DataType::Float32:
                        apply(reinterpret_cast<float*>(bytes.data()), count);
                        break;
                    case Core::DataType::Float64:
                        apply(reinterpret_cast<double*>(bytes.data()), count);
                        break;
                    case Core::DataType::Complex32:
                        apply(reinterpret_cast<float*>(bytes.data()), count * 2);
                        break;
                    case Core::DataType::Complex64:
                        apply(reinterpret_cast<double*>(bytes.data()), count * 2);
                        break;
                    case Core::DataType::Int32:
                        apply(reinterpret_cast<std::int32_t*>(bytes.data()), count);
                        break;
                    case Core::DataType::Int64:
                        apply(reinterpret_cast<std::int64_t*>(bytes.data()), count);
                        break;
                }
                return bytes;
            }

            /**
             * @brief Returns a registry whose kernel checks the fingerprint, binds the views and
             *        transforms the output pack in place after copying the input into it.
             * @param layout The layout the kernel binds; its fingerprint names the operation.
             */
            SushiRuntime::sushi_ptr<KernelRegistry> transform_kernels(
                std::shared_ptr<const PackLayout> layout, std::atomic<int>& runs)
            {
                auto kernels = SushiRuntime::make_sushi<KernelRegistry>();
                const SushiRuntime::OperationID operation =
                    remote_operation(K_OPERATION_NAME, layout->fingerprint());
                kernels->register_kernel(
                    operation,
                    [layout, operation, &runs](sycl::queue& queue,
                                               SushiRuntime::span<const BufferView> inputs,
                                               SushiRuntime::span<const BufferView> outputs,
                                               const SushiRuntime::NodeOptions& options)
                        -> sycl::event
                    {
                        SB_THROW_IF(options.operation != operation,
                                    "the task's fingerprint is not the kernel's");
                        const std::vector<BoundPack> in =
                            bind_views(inputs,
                                       SushiRuntime::span<const PackLayout>(layout.get(), 1));
                        const std::vector<BoundPack> out =
                            bind_views(outputs,
                                       SushiRuntime::span<const PackLayout>(layout.get(), 1));
                        runs.fetch_add(1, std::memory_order_relaxed);

                        std::vector<sycl::event> previous;
                        previous.push_back(queue.memcpy(outputs[0].data, inputs[0].data,
                                                        layout->size_bytes()));
                        for (std::size_t type = 0; type < K_PACK_SEGMENT_COUNT; ++type)
                        {
                            const PackSpan segment =
                                out[0].segment(static_cast<Core::DataType>(type));
                            if (segment.count == 0)
                                continue;
                            previous.assign(1, submit_segment(queue, segment, previous));
                        }
                        (void)in;
                        return previous.back();
                    });
                return kernels;
            }

            /** @brief Holds what one run of the round trip saw. */
            struct RoundTrip
            {
                bool worker_joined = false; /**< The worker connected before the run. */
                bool worker_has_op = false; /**< The master saw the worker advertise the op. */
                int local_runs = 0;         /**< Runs of the master's own kernel. */
                int worker_runs = 0;        /**< Runs of the worker's kernel. */
                std::uint64_t offloaded = 0; /**< Dispatches the master sent to the worker. */
                bool bitwise_equal = false; /**< Every result tensor equals its expected bytes. */
            };

            /**
             * @brief Packs one tensor per type, runs the transform as a registered task and
             *        unpacks the result.
             * @param worker_layout_differs The worker registers for a pack without its last tensor.
             */
            RoundTrip run_round_trip(bool worker_layout_differs)
            {
                std::atomic<int> local_runs{0};
                std::atomic<int> worker_runs{0};

                SushiRuntime::Runtime master = SushiRuntime::Runtime::create().value();
                std::vector<Tensor> sources = one_tensor_per_type(master);
                std::vector<Tensor> results = one_tensor_per_type(master);
                fill(sources);
                const auto layout = std::make_shared<const PackLayout>(
                    PackLayout::of(
                        SushiRuntime::span<const Tensor>(sources.data(), sources.size())));

                std::shared_ptr<const PackLayout> worker_layout = layout;
                if (worker_layout_differs)
                {
                    std::vector<Tensor> other = sources;
                    other.pop_back();
                    worker_layout = std::make_shared<const PackLayout>(
                        PackLayout::of(
                            SushiRuntime::span<const Tensor>(other.data(), other.size())));
                }

                InProcessFabric fabric;
                SushiRuntime::Runtime worker_runtime = SushiRuntime::Runtime::create().value();
                WorkerAgent worker(worker_runtime, InProcessTransport::make_worker(fabric),
                                   transform_kernels(worker_layout, worker_runs));

                const SushiRuntime::sushi_ptr<KernelRegistry> kernels =
                    transform_kernels(layout, local_runs);
                ClusterOptions options;
                options.policy =
                    std::make_shared<SushiRuntime::Distributed::LeastLoadedOffloadPolicy>(0);
                options.kernels = kernels;
                Cluster cluster = Cluster::attach(master, InProcessTransport::make_master(fabric),
                                                  std::move(options)).value();
                RoundTrip trip;
                trip.worker_joined = worker.start().is_ok() && cluster.wait_for_workers(1, K_WAIT);

                ClusterBinding binding(cluster, kernels);
                const TensorPack input(master, *layout);
                const TensorPack output(master, *layout);
                binding.share(input);
                binding.share(output);

                SushiRuntime::NodeOptions task;
                task.name = "sushiblas.test.pack_round_trip";
                task.operation = remote_operation(K_OPERATION_NAME, layout->fingerprint());
                trip.worker_has_op = binding.has_worker_for(task.operation);

                ScratchPool scratch(master);
                const TensorPack* inputs[] = {&input};
                const TensorPack* outputs[] = {&output};
                auto graph = master.graph();
                graph.extend([&](SushiRuntime::Recorder& recorder)
                {
                    TaskRecorder tensors(recorder, scratch);
                    (void)input.record_gather(tensors, SushiRuntime::span<const Tensor>(
                                                           sources.data(), sources.size()));
                    (void)record_remote(recorder, binding, task,
                                        SushiRuntime::span<const TensorPack* const>(inputs, 1),
                                        SushiRuntime::span<const TensorPack* const>(outputs, 1));
                    (void)output.record_scatter(tensors, SushiRuntime::span<const Tensor>(
                                                             results.data(), results.size()));
                });
                graph.run().value();

                trip.bitwise_equal = true;
                for (std::size_t index = 0; index < results.size(); ++index)
                {
                    const std::vector<std::uint8_t> expected = expected_bytes(sources[index]);
                    trip.bitwise_equal = trip.bitwise_equal &&
                                         std::memcmp(expected.data(), results[index].data(),
                                                     expected.size()) == 0;
                }
                trip.local_runs = local_runs.load(std::memory_order_relaxed);
                trip.worker_runs = worker_runs.load(std::memory_order_relaxed);
                trip.offloaded = cluster.offloaded();
                worker.stop();
                return trip;
            }
        } // namespace

        /** @brief Tests that a pack of all seven types returns from the worker bit for bit. */
        TEST(Medium_RemoteTask, PackOfEveryTypeRoundTripsThroughAWorker)
        {
            if (sycl::device::get_devices().empty())
                GTEST_SKIP() << "SYCL reports no device.";

            const RoundTrip trip = run_round_trip(false);

            ASSERT_TRUE(trip.worker_joined);
            EXPECT_TRUE(trip.worker_has_op);
            EXPECT_EQ(trip.worker_runs, 1);
            EXPECT_EQ(trip.local_runs, 0);
            EXPECT_EQ(trip.offloaded, 1U);
            EXPECT_TRUE(trip.bitwise_equal);
        }

        /**
         * @brief Tests that a worker whose layout differs advertises another operation, so the
         *        master runs the task itself.
         */
        TEST(Medium_RemoteTask, WorkerOfAnotherLayoutNeverSeesTheTask)
        {
            if (sycl::device::get_devices().empty())
                GTEST_SKIP() << "SYCL reports no device.";

            const RoundTrip trip = run_round_trip(true);

            ASSERT_TRUE(trip.worker_joined);
            EXPECT_FALSE(trip.worker_has_op);
            EXPECT_EQ(trip.worker_runs, 0);
            EXPECT_EQ(trip.local_runs, 1);
            EXPECT_EQ(trip.offloaded, 0U);
            EXPECT_TRUE(trip.bitwise_equal);
        }
    } // namespace Test
} // namespace SushiBLAS
