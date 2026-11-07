/**************************************************************************/
/* test_sparse_vector_operand.cpp                                         */
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
 * @file test_sparse_vector_operand.cpp
 * @brief Tests how a compressed vector is described and the sizes that are refused, on arrays
 *        built by hand so no runtime runs.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/sparse/sparse_vector_operand.hpp>
#include "support/hand_built_operand.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The made-up allocation of each array of the vector.
            constexpr std::size_t K_INDICES_ALLOCATION = 1;
            constexpr std::size_t K_VALUES_ALLOCATION = 2;

            /**
             * @brief Holds the two arrays of a zero-based vector with INT64 indices, which a test
             *        pairs with a size.
             */
            struct VectorArrays
            {
                /** @brief Builds the arrays of a vector storing entries entries. */
                explicit VectorArrays(int64_t entries)
                    : indices({entries}, Core::DataType::Int64, K_INDICES_ALLOCATION),
                      values({entries}, Core::DataType::Float32, K_VALUES_ALLOCATION)
                {

                }

                HandBuiltOperand indices; /**< The stored positions. */
                HandBuiltOperand values;  /**< The stored values. */

                /** @brief Returns the arrays paired with the size of a vector. */
                [[nodiscard]] Dispatch::PlacedSparseVector placed(int64_t size) const
                {
                    return Dispatch::PlacedSparseVector{size, Core::IndexBase::Zero,
                                                        indices.placed(), values.placed()};
                }
            };
        } // namespace

        TEST(Small_SparseVectorOperand, DescribesItsSizeEntriesAndFirstIndex)
        {
            const VectorArrays arrays(3);

            const Dispatch::SparseVectorOperand v =
                Dispatch::describe_sparse_vector(arrays.placed(10), "op", "x");

            EXPECT_EQ(v.size, int64_t{10});
            EXPECT_EQ(v.nnz, int64_t{3});
            EXPECT_EQ(v.first_index, int64_t{0});
        }

        TEST(Small_SparseVectorOperand, RefusesANegativeSize)
        {
            const VectorArrays arrays(0);

            EXPECT_THROW((void)Dispatch::describe_sparse_vector(arrays.placed(-1), "op", "x"),
                         std::invalid_argument);
        }

        TEST(Small_SparseVectorOperand, RefusesMoreEntriesThanPositions)
        {
            const VectorArrays arrays(3);

            EXPECT_THROW((void)Dispatch::describe_sparse_vector(arrays.placed(2), "op", "x"),
                         std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
