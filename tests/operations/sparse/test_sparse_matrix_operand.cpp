/**************************************************************************/
/* test_sparse_matrix_operand.cpp                                         */
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
 * @file test_sparse_matrix_operand.cpp
 * @brief Tests how a CSR matrix is described and the shapes of its arrays that are refused, on
 *        arrays built by hand so no runtime runs.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/sparse/sparse_matrix_operand.hpp>
#include "support/hand_built_operand.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The made-up allocation of each array of the matrix.
            constexpr std::size_t K_OFFSETS_ALLOCATION = 1;
            constexpr std::size_t K_COLUMNS_ALLOCATION = 2;
            constexpr std::size_t K_VALUES_ALLOCATION = 3;

            /**
             * @brief Holds the three arrays of a 3 x 4 one-based matrix of five entries with INT32
             *        indices, which a test spoils one array at a time.
             */
            struct MatrixArrays
            {
                HandBuiltOperand row_offsets{{4}, Core::DataType::Int32, K_OFFSETS_ALLOCATION};
                HandBuiltOperand column_indices{{5}, Core::DataType::Int32, K_COLUMNS_ALLOCATION};
                HandBuiltOperand values{{5}, Core::DataType::Float32, K_VALUES_ALLOCATION};

                /** @brief Returns the arrays paired with the extents and the index base of a matrix. */
                [[nodiscard]] Dispatch::PlacedSparseMatrix placed() const
                {
                    return Dispatch::PlacedSparseMatrix{3,
                                                        4,
                                                        Core::IndexBase::One,
                                                        row_offsets.placed(),
                                                        column_indices.placed(),
                                                        values.placed()};
                }
            };
        } // namespace

        TEST(Small_SparseMatrixOperand, DescribesItsExtentsEntriesAndFirstIndex)
        {
            const MatrixArrays arrays;

            const Dispatch::SparseMatrixOperand m =
                Dispatch::describe_sparse_matrix(arrays.placed(), "op", "A");

            EXPECT_EQ(m.rows, int64_t{3});
            EXPECT_EQ(m.cols, int64_t{4});
            EXPECT_EQ(m.nnz, int64_t{5});
            EXPECT_EQ(m.first_index, int64_t{1});
        }

        TEST(Small_SparseMatrixOperand, RefusesAWrongOffsetCount)
        {
            MatrixArrays arrays;
            arrays.row_offsets = HandBuiltOperand({3}, Core::DataType::Int32, K_OFFSETS_ALLOCATION);

            EXPECT_THROW((void)Dispatch::describe_sparse_matrix(arrays.placed(), "op", "A"),
                         std::invalid_argument);
        }

        TEST(Small_SparseMatrixOperand, RefusesMixedIndexDtypes)
        {
            MatrixArrays arrays;
            arrays.column_indices =
                HandBuiltOperand({5}, Core::DataType::Int64, K_COLUMNS_ALLOCATION);

            EXPECT_THROW((void)Dispatch::describe_sparse_matrix(arrays.placed(), "op", "A"),
                         std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
