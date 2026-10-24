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
            /** @brief Returns a shared tensor of a type with count elements. */
            Tensor shared_tensor(SushiRuntime::Runtime& runtime, Core::DataType dtype,
                                 std::int64_t count)
            {
                const std::int64_t dims[] = {count};
                const std::size_t bytes =
                    static_cast<std::size_t>(count) * Core::bytes_per_element(dtype);
                return Tensor(SushiRuntime::make_sushi<Storage>(runtime, bytes,
                                                                SushiRuntime::Residency::Shared, 0),
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
                                               sources.data(), sources.size())));
            ScratchPool scratch(runtime);
            auto graph = runtime.graph();
            graph.extend([&](SushiRuntime::Recorder& recorder)
            {
                TaskRecorder tensors(recorder, scratch);
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
                                               layout_tensors.data(), layout_tensors.size())));
            ScratchPool scratch(runtime);
            auto graph = runtime.graph();
            graph.extend([&](SushiRuntime::Recorder& recorder)
            {
                TaskRecorder tensors(recorder, scratch);
                EXPECT_THROW((void)pack.record_gather(tensors, SushiRuntime::span<const Tensor>(
                                                                   other.data(), other.size())),
                             std::runtime_error);
            });
        }
    } // namespace Test
} // namespace SushiBLAS
