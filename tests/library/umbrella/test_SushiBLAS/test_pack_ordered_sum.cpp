/**************************************************************************/
/* test_pack_ordered_sum.cpp                                              */
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
 * @file test_pack_ordered_sum.cpp
 * @brief Tests that ordered_sum sums the same segment tensor of several packs into another pack.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiBLAS/cluster/tensor_pack.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/elementwise/elementwise.hpp>
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
        } // namespace

        /** @brief Tests that ordered_sum over segment tensors of three packs fills a fourth. */
        TEST(Medium_PackOrderedSum, SumsTheSameSegmentOfThreePacksIntoAFourth)
        {
            if (sycl::device::get_devices().empty())
                GTEST_SKIP() << "SYCL reports no device.";

            SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
            const std::vector<Tensor> shape_source = {
                shared_tensor(runtime, Core::DataType::Float32, 4),
                shared_tensor(runtime, Core::DataType::Int32, 2),
                shared_tensor(runtime, Core::DataType::Float32, 3)};
            const PackLayout layout = PackLayout::of(
                SushiRuntime::span<const Tensor>(shape_source.data(), shape_source.size()));
            std::vector<TensorPack> replicas;
            replicas.reserve(4);
            for (int replica = 0; replica < 4; ++replica)
                replicas.emplace_back(runtime, layout);

            std::vector<Tensor> inputs;
            for (int replica = 0; replica < 3; ++replica)
            {
                inputs.push_back(replicas[replica].segment_tensor(Core::DataType::Float32));
                auto* values = inputs.back().data_as<float>();
                for (std::int64_t element = 0; element < inputs.back().num_elements; ++element)
                    values[element] = static_cast<float>((replica + 1) * 10 + element);
            }
            Tensor result = replicas[3].segment_tensor(Core::DataType::Float32);
            std::memset(result.data(), 0, static_cast<std::size_t>(result.num_elements) * sizeof(float));

            ScratchPool scratch(runtime);
            auto graph = runtime.graph();
            graph.extend([&](SushiRuntime::Recorder& recorder)
            {
                TaskRecorder tensors(recorder, scratch);
                const std::array<double, 3> weights = {1.0, 1.0, 1.0};
                (void)ElementwiseOps(tensors).ordered_sum(
                    SushiRuntime::span<const Tensor>(inputs.data(), inputs.size()),
                    SushiRuntime::span<const double>(weights.data(), weights.size()), result);
            });
            graph.run().value();

            ASSERT_EQ(result.num_elements, 7);
            for (std::int64_t element = 0; element < result.num_elements; ++element)
                EXPECT_EQ(result.data_as<float>()[element], static_cast<float>(60 + 3 * element))
                    << "element " << element;
        }

    } // namespace Test
} // namespace SushiBLAS
