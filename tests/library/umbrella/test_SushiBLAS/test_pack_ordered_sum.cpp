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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file test_pack_ordered_sum.cpp
 * @brief Tests that ordered_sum sums the same segment tensor of several packs into another pack.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/tensor/runtime_block.hpp>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiBLAS/cluster/tensor_pack.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Holds the dtype and the element count of one tensor a pack is laid out from. */
            struct EntryShape
            {
                /** @brief The dtype of the entry. */
                Core::DataType dtype;

                /** @brief The element count of the entry. */
                std::int64_t count;
            };

            /** @brief Returns the entries every pack of the test is laid out from. */
            std::vector<EntryShape> pack_entries()
            {
                return {{Core::DataType::Float32, 4},
                        {Core::DataType::Int32, 2},
                        {Core::DataType::Float32, 3}};
            }

            /** @brief Returns a shared tensor of a type with count elements. */
            Tensor shared_tensor(Core::DataType dtype, std::int64_t count)
            {
                const std::int64_t dims[] = {count};
                const std::size_t bytes =
                    static_cast<std::size_t>(count) * Core::bytes_per_element(dtype);
                return Tensor(make_runtime_storage(TestEnvironment::runtime(), bytes,
                                                   SushiRuntime::Residency::Shared, 0),
                              SushiRuntime::span<const std::int64_t>(dims, 1), dtype);
            }

            /** @brief Returns count values that start at first and rise by one. */
            std::vector<float> ramp(float first, std::int64_t count)
            {
                std::vector<float> values(static_cast<std::size_t>(count));
                for (std::int64_t element = 0; element < count; ++element)
                    values[static_cast<std::size_t>(element)] = first + static_cast<float>(element);
                return values;
            }

            /** @brief Returns the sum of the ramps that start at 10, 20 and 30: 60 plus three per element. */
            std::vector<float> sum_of_the_three_ramps(std::int64_t count)
            {
                std::vector<float> values(static_cast<std::size_t>(count));
                for (std::int64_t element = 0; element < count; ++element)
                    values[static_cast<std::size_t>(element)] = 60.0f + 3.0f * static_cast<float>(element);
                return values;
            }
        } // namespace

        /** @brief Builds four packs over one layout, and sums the FLOAT32 segment of three into the fourth. */
        class Medium_PackOrderedSum : public SushiBLASTest
        {
            protected:
                /** @brief Returns the layout of the entries of pack_entries. */
                static PackLayout layout_of_entries()
                {
                    std::vector<Tensor> shape_source;
                    for (const EntryShape& entry : pack_entries())
                        shape_source.push_back(shared_tensor(entry.dtype, entry.count));
                    return PackLayout::of(
                        SushiRuntime::span<const Tensor>(shape_source.data(), shape_source.size()));
                }

                /**
                 * @brief Fills three packs, appends them to packs, and returns their FLOAT32 segments.
                 *
                 * The segment of pack r starts at 10 * (r + 1) and rises by one per element.
                 */
                static std::vector<Tensor> filled_segments(std::vector<TensorPack>& packs)
                {
                    std::vector<Tensor> segments;
                    for (int replica = 0; replica < 3; ++replica)
                    {
                        packs.push_back(make_pack());
                        segments.push_back(packs.back().segment_tensor(Core::DataType::Float32));
                        fill_tensor(segments.back(),
                                    ramp(static_cast<float>((replica + 1) * 10),
                                         segments.back().num_elements));
                    }
                    return segments;
                }

                /** @brief Returns a pack over the layout of the entries. */
                static TensorPack make_pack()
                {
                    return TensorPack(TestEnvironment::runtime(), layout_of_entries(), 0);
                }
        };

        /** @brief Tests that ordered_sum over segment tensors of three packs fills a fourth. */
        TEST_F(Medium_PackOrderedSum, SumsTheSameSegmentOfThreePacksIntoAFourth)
        {
            std::vector<TensorPack> sources;
            const std::vector<Tensor> inputs = filled_segments(sources);
            TensorPack target = make_pack();
            Tensor result = target.segment_tensor(Core::DataType::Float32);
            fill_tensor(result, std::vector<float>(static_cast<std::size_t>(result.num_elements), 0.0f));

            const std::vector<double> weights = {1.0, 1.0, 1.0};
            (void)ElementwiseOps(recorder()).ordered_sum(
                SushiRuntime::span<const Tensor>(inputs.data(), inputs.size()),
                SushiRuntime::span<const double>(weights.data(), weights.size()), result);
            recording->run().value();

            ASSERT_EQ(result.num_elements, 7);
            expect_tensor_equal(result, sum_of_the_three_ramps(result.num_elements));
        }
    } // namespace Test
} // namespace SushiBLAS
