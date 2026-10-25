/**************************************************************************/
/* test_tensor_pack.cpp                                                   */
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
 * @file test_tensor_pack.cpp
 * @brief Tests that TensorPack gathers tensors into its buffer and scatters them back unchanged.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiBLAS/cluster/tensor_pack.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns a shared tensor of a type with count elements on one device. */
            Tensor shared_tensor(SushiRuntime::Runtime& runtime, Core::DataType dtype,
                                 std::int64_t count, std::size_t device_index = 0)
            {
                const std::int64_t dims[] = {count};
                const std::size_t bytes =
                    static_cast<std::size_t>(count) * Core::bytes_per_element(dtype);
                return Tensor(SushiRuntime::make_sushi<Storage>(runtime, bytes,
                                                                SushiRuntime::Residency::Shared,
                                                                device_index),
                              SushiRuntime::span<const std::int64_t>(dims, 1), dtype);
            }

            /** @brief Returns the byte count of a tensor's elements. */
            std::size_t byte_count(const Tensor& tensor)
            {
                return static_cast<std::size_t>(tensor.num_elements) *
                       Core::bytes_per_element(tensor.dtype);
            }
        } // namespace

        /** @brief Tests that a gather then a scatter copies every tensor's bytes unchanged. */
        TEST(Medium_TensorPack, GatherThenScatterCopiesEveryByte)
        {
            if (sycl::device::get_devices().empty())
                GTEST_SKIP() << "SYCL reports no device.";

            SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
            std::vector<Tensor> sources = {shared_tensor(runtime, Core::DataType::Float64, 9),
                                           shared_tensor(runtime, Core::DataType::Half, 5),
                                           shared_tensor(runtime, Core::DataType::Int32, 3)};
            std::vector<Tensor> targets = {shared_tensor(runtime, Core::DataType::Float64, 9),
                                           shared_tensor(runtime, Core::DataType::Half, 5),
                                           shared_tensor(runtime, Core::DataType::Int32, 3)};
            for (std::size_t index = 0; index < sources.size(); ++index)
            {
                auto* bytes = static_cast<std::uint8_t*>(sources[index].data());
                for (std::size_t offset = 0; offset < byte_count(sources[index]); ++offset)
                    bytes[offset] = static_cast<std::uint8_t>(offset * 31 + index * 7 + 1);
                std::memset(targets[index].data(), 0, byte_count(targets[index]));
            }

            const TensorPack pack(runtime, PackLayout::of(SushiRuntime::span<const Tensor>(
                                               sources.data(), sources.size())), 0);
            ScratchPool scratch(runtime);
            LateCountHold late_counts(runtime);
            auto graph = runtime.graph();
            graph.extend([&](SushiRuntime::Recorder& recorder)
            {
                TaskRecorder tensors(recorder, scratch, late_counts);
                (void)pack.record_gather(tensors, SushiRuntime::span<const Tensor>(
                                                      sources.data(), sources.size()));
                (void)pack.record_scatter(tensors, SushiRuntime::span<const Tensor>(
                                                       targets.data(), targets.size()));
            });
            graph.run().value();

            for (std::size_t index = 0; index < sources.size(); ++index)
                EXPECT_EQ(std::memcmp(sources[index].data(), targets[index].data(),
                                      byte_count(sources[index])),
                          0)
                    << "tensor " << index;
        }

        /** @brief Tests that the tensor of an entry reads the bytes the gather wrote. */
        TEST(Medium_TensorPack, EntryTensorReadsTheBytesTheGatherWrote)
        {
            if (sycl::device::get_devices().empty())
                GTEST_SKIP() << "SYCL reports no device.";

            SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
            std::vector<Tensor> sources = {shared_tensor(runtime, Core::DataType::Float64, 9),
                                           shared_tensor(runtime, Core::DataType::Half, 5),
                                           shared_tensor(runtime, Core::DataType::Float64, 2),
                                           shared_tensor(runtime, Core::DataType::Int32, 3)};
            for (std::size_t index = 0; index < sources.size(); ++index)
            {
                auto* bytes = static_cast<std::uint8_t*>(sources[index].data());
                for (std::size_t offset = 0; offset < byte_count(sources[index]); ++offset)
                    bytes[offset] = static_cast<std::uint8_t>(offset * 17 + index * 5 + 3);
            }

            std::vector<Tensor> views;
            {
                const TensorPack pack(runtime, PackLayout::of(SushiRuntime::span<const Tensor>(
                                                   sources.data(), sources.size())), 0);
                ScratchPool scratch(runtime);
                LateCountHold late_counts(runtime);
                auto graph = runtime.graph();
                graph.extend([&](SushiRuntime::Recorder& recorder)
                {
                    TaskRecorder tensors(recorder, scratch, late_counts);
                    (void)pack.record_gather(tensors, SushiRuntime::span<const Tensor>(
                                                          sources.data(), sources.size()));
                });
                graph.run().value();

                for (std::size_t index = 0; index < sources.size(); ++index)
                    views.push_back(pack.tensor(index));
                EXPECT_THROW((void)pack.tensor(sources.size()), std::out_of_range);
            }

            // The pack is gone; the views hold its storage.
            for (std::size_t index = 0; index < sources.size(); ++index)
            {
                EXPECT_EQ(views[index].dtype, sources[index].dtype) << "entry " << index;
                EXPECT_EQ(views[index].num_elements, sources[index].num_elements)
                    << "entry " << index;
                EXPECT_EQ(views[index].shape[0], sources[index].shape[0]) << "entry " << index;
                EXPECT_EQ(std::memcmp(views[index].data(), sources[index].data(),
                                      byte_count(sources[index])),
                          0)
                    << "entry " << index;
            }
        }

        /** @brief Tests that the segment tensor of each type has the segment's size and offset. */
        TEST(Medium_TensorPack, SegmentTensorHasTheSegmentsSizeAndOffset)
        {
            if (sycl::device::get_devices().empty())
                GTEST_SKIP() << "SYCL reports no device.";

            SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
            const std::array<Core::DataType, 8> types = {
                Core::DataType::UInt32,    Core::DataType::Int64,     Core::DataType::Int32,
                Core::DataType::Complex64, Core::DataType::Complex32, Core::DataType::Float64,
                Core::DataType::Float32,   Core::DataType::Half};
            std::vector<Tensor> sources;
            for (std::size_t index = 0; index < types.size(); ++index)
            {
                sources.push_back(shared_tensor(runtime, types[index],
                                                static_cast<std::int64_t>(index + 1)));
                sources.push_back(shared_tensor(runtime, types[index], 2));
            }
            const TensorPack pack(runtime, PackLayout::of(SushiRuntime::span<const Tensor>(
                                               sources.data(), sources.size())), 0);

            for (const Core::DataType dtype : types)
            {
                const PackSegment& segment = pack.layout().segment(dtype);
                const Tensor view = pack.segment_tensor(dtype);
                EXPECT_EQ(view.rank, 1);
                EXPECT_EQ(view.dtype, dtype);
                EXPECT_EQ(static_cast<std::size_t>(view.num_elements), segment.element_count);
                EXPECT_EQ(static_cast<std::size_t>(view.num_elements) *
                              Core::bytes_per_element(dtype),
                          segment.size_bytes);
                EXPECT_EQ(static_cast<const std::byte*>(view.data()),
                          static_cast<const std::byte*>(pack.data()) + segment.offset_bytes);
            }
        }

        /** @brief Tests that a gather refuses a tensor whose shape differs from its entry. */
        TEST(Medium_TensorPack, GatherRefusesATensorOfAnotherShape)
        {
            if (sycl::device::get_devices().empty())
                GTEST_SKIP() << "SYCL reports no device.";

            SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
            const std::vector<Tensor> layout_tensors = {
                shared_tensor(runtime, Core::DataType::Float32, 4)};
            const std::vector<Tensor> other = {shared_tensor(runtime, Core::DataType::Float32, 5)};
            const TensorPack pack(runtime, PackLayout::of(SushiRuntime::span<const Tensor>(
                                               layout_tensors.data(), layout_tensors.size())), 0);
            ScratchPool scratch(runtime);
            LateCountHold late_counts(runtime);
            auto graph = runtime.graph();
            graph.extend([&](SushiRuntime::Recorder& recorder)
            {
                TaskRecorder tensors(recorder, scratch, late_counts);
                EXPECT_THROW((void)pack.record_gather(tensors, SushiRuntime::span<const Tensor>(
                                                                   other.data(), other.size())),
                             std::runtime_error);
            });
        }

        /** @brief Tests that a pack allocates on the device it is given, not on device 0. */
        TEST(Medium_Regression_TensorPack, BufferLivesOnTheDeviceItIsGiven)
        {
            if (sycl::device::get_devices().empty())
                GTEST_SKIP() << "SYCL reports no device.";

            SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
            const std::size_t device_count = runtime.advanced().device_count();
            for (std::size_t device = 0; device < device_count; ++device)
            {
                const std::vector<Tensor> shapes = {
                    shared_tensor(runtime, Core::DataType::Float32, 4, device)};
                const TensorPack pack(runtime, PackLayout::of(SushiRuntime::span<const Tensor>(
                                                   shapes.data(), shapes.size())), device);
                EXPECT_EQ(pack.device_index(), device);
                EXPECT_EQ(pack.tensor(0).storage->device_index(), device);
                EXPECT_EQ(pack.tensor(0).storage->residency(), SushiRuntime::Residency::Host);
            }

            const std::vector<Tensor> shapes = {shared_tensor(runtime, Core::DataType::Float32, 4)};
            EXPECT_THROW(TensorPack(runtime, PackLayout::of(SushiRuntime::span<const Tensor>(
                                                 shapes.data(), shapes.size())), device_count),
                         std::runtime_error);
        }

        /** @brief Tests that a gather refuses a tensor on another device than the pack. */
        TEST(Medium_Regression_TensorPack, GatherRefusesATensorOnAnotherDevice)
        {
            if (sycl::device::get_devices().empty())
                GTEST_SKIP() << "SYCL reports no device.";

            SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
            if (runtime.advanced().device_count() < 2)
                GTEST_SKIP() << "The runtime drives one device; two are needed.";

            const std::vector<Tensor> on_second = {
                shared_tensor(runtime, Core::DataType::Float32, 4, 1)};
            const TensorPack pack(runtime, PackLayout::of(SushiRuntime::span<const Tensor>(
                                               on_second.data(), on_second.size())), 0);
            ScratchPool scratch(runtime);
            LateCountHold late_counts(runtime);
            auto graph = runtime.graph();
            graph.extend([&](SushiRuntime::Recorder& recorder)
            {
                TaskRecorder tensors(recorder, scratch, late_counts);
                EXPECT_THROW((void)pack.record_gather(tensors, SushiRuntime::span<const Tensor>(
                                                                   on_second.data(),
                                                                   on_second.size())),
                             std::runtime_error);
            });
        }
    } // namespace Test
} // namespace SushiBLAS
