/**************************************************************************/
/* test_index_operands.cpp                                                */
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
 * @file test_index_operands.cpp
 * @brief Tests that the record-time checks of the row index operations accept a fit operand and refuse a misfit one.
 * @author Mustafa Garip
 */

#include <array>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/gather/index_operands.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Gives each check a host engine to take its tensors from. */
        class Small_IndexOperands : public HostEngineTest
        {
            protected:
                /** @brief Returns a view of t with the given extents and strides, from its first element. */
                static Tensor strided_view(const Tensor& t, std::array<int64_t, 2> extents,
                                           std::array<int64_t, 2> element_strides)
                {
                    std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
                    strides[0] = element_strides[0];
                    strides[1] = element_strides[1];
                    const std::array<int64_t, 2> dims = extents;
                    return Tensor(t.storage,
                                  SushiRuntime::span<const int64_t>(dims.data(), dims.size()),
                                  strides, t.dtype, 0, t.layout);
                }
        };

        TEST_F(Small_IndexOperands, AnIndexIsAcceptedInEitherIntegerWidthAndRefusedAsFloat)
        {
            const Tensor narrow = engine->create_tensor({4}, Core::DataType::Int32);
            const Tensor wide = engine->create_tensor({4}, Core::DataType::Int64);
            const Tensor floating = engine->create_tensor({4}, Core::DataType::Float32);

            EXPECT_NO_THROW(Gather::require_index_operand(narrow, "op", "index"));
            EXPECT_NO_THROW(Gather::require_index_operand(wide, "op", "index"));
            EXPECT_THROW(Gather::require_index_operand(floating, "op", "index"),
                         std::invalid_argument);
        }

        TEST_F(Small_IndexOperands, AnIndexOfRankTwoIsRefused)
        {
            const Tensor index = engine->create_tensor({2, 2}, Core::DataType::Int64);

            EXPECT_THROW(Gather::require_index_operand(index, "op", "index"),
                         std::invalid_argument);
        }

        TEST_F(Small_IndexOperands, AnIndexWithoutStorageIsRefused)
        {
            const int64_t dims[1] = {3};
            const Tensor index = Tensor::make_shape_only(SushiRuntime::span<const int64_t>(dims, 1),
                                                         Core::DataType::Int64);

            EXPECT_THROW(Gather::require_index_operand(index, "op", "index"),
                         std::invalid_argument);
        }

        TEST_F(Small_IndexOperands, ADataOperandIsAcceptedOnlyAtTheNamedRank)
        {
            const Tensor matrix = engine->create_tensor({3, 4}, Core::DataType::Float32);
            const Tensor vector = engine->create_tensor({12}, Core::DataType::Float32);

            EXPECT_NO_THROW(Gather::require_data_operand(matrix, 2, "op", "table"));
            EXPECT_THROW(Gather::require_data_operand(vector, 2, "op", "table"),
                         std::invalid_argument);
        }

        TEST_F(Small_IndexOperands, ADenseMatrixHasDistinctElementsAndAZeroRowStrideDoesNot)
        {
            const Tensor matrix = engine->create_tensor({3, 4}, Core::DataType::Float32);
            const Tensor repeated = strided_view(matrix, {3, 4}, {0, 1});

            EXPECT_NO_THROW(Gather::require_distinct_elements(matrix, "op", "rows"));
            EXPECT_THROW(Gather::require_distinct_elements(repeated, "op", "rows"),
                         std::invalid_argument);
        }

        TEST_F(Small_IndexOperands, AViewWhoseAxesInterleaveIsRefused)
        {
            const Tensor matrix = engine->create_tensor({3, 4}, Core::DataType::Float32);
            const Tensor interleaved = strided_view(matrix, {3, 3}, {1, 1});

            EXPECT_THROW(Gather::require_distinct_elements(interleaved, "op", "rows"),
                         std::invalid_argument);
        }

        TEST_F(Small_IndexOperands, TwoRowRangesAreDisjointOnlyWhenTheyShareNoElement)
        {
            const Tensor matrix = engine->create_tensor({4, 4}, Core::DataType::Float32);
            const Tensor top = matrix.slice(0, 0, 2);
            const Tensor overlapping = matrix.slice(0, 1, 3);
            const Tensor bottom = matrix.slice(0, 2, 4);

            EXPECT_NO_THROW(Gather::require_disjoint(top, bottom, "op", "rows", "table"));
            EXPECT_THROW(Gather::require_disjoint(top, overlapping, "op", "rows", "table"),
                         std::invalid_argument);
        }

        TEST_F(Small_IndexOperands, ADenseMatrixSpansAllItsElements)
        {
            const Tensor matrix = engine->create_tensor({3, 4}, Core::DataType::Float32);

            EXPECT_EQ(Gather::spanned_elements(matrix), 12U);
        }

        TEST_F(Small_IndexOperands, AColumnSliceSpansFromItsFirstToItsLastElement)
        {
            const Tensor matrix = engine->create_tensor({3, 4}, Core::DataType::Float32);
            const Tensor columns = matrix.slice(1, 1, 3);

            // The last element is at offset 2 * 4 + 1 * 1, so the span is that plus one.
            EXPECT_EQ(Gather::spanned_elements(columns), 10U);
        }

        TEST_F(Small_IndexOperands, AnEmptyMatrixSpansNothing)
        {
            const Tensor empty = engine->create_tensor({0, 4}, Core::DataType::Float32);

            EXPECT_EQ(Gather::spanned_elements(empty), 0U);
        }
    } // namespace Test
} // namespace SushiBLAS
