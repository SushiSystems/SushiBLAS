/**************************************************************************/
/* test_sparse_matrix.cpp                                                 */
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
 * @file test_sparse_matrix.cpp
 * @brief Tests that a SparseMatrix reads its entry count and both dtypes from its tensors.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/sparse_storage/sparse_matrix.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Returns a rank-one tensor of count elements of dtype, with no storage.
        static Tensor vector_of(int64_t count, Core::DataType dtype)
        {
            const int64_t dims[1] = {count};
            return Tensor::make_shape_only(SushiRuntime::span<const int64_t>(dims, 1), dtype);
        }

        TEST(Small_SparseMatrix, ADefaultMatrixIsEmpty)
        {
            const SparseMatrix matrix;

            EXPECT_EQ(matrix.rows, int64_t{0});
            EXPECT_EQ(matrix.cols, int64_t{0});
            EXPECT_EQ(matrix.nnz(), int64_t{0});
        }

        TEST(Small_SparseMatrix, ADefaultMatrixIsZeroBased)
        {
            const SparseMatrix matrix;

            EXPECT_EQ(matrix.index_base, Core::IndexBase::Zero);
        }

        namespace
        {
            /** @brief Holds the dtypes of the three arrays of one matrix. */
            struct DtypeRow
            {
                const char* name;               /**< The row suffix. */
                Core::DataType offset_dtype;    /**< The element type of the row offsets. */
                Core::DataType column_dtype;    /**< The element type of the column indices. */
                Core::DataType value_dtype;     /**< The element type of the values. */
            };

            /** @brief Returns the dtype triples the matrix is read with. */
            std::vector<DtypeRow> dtype_rows()
            {
                return {{"Int32IndicesComplex64Values", Core::DataType::Int32,
                         Core::DataType::Int32, Core::DataType::Complex64},
                        {"Int64OffsetsInt32ColumnsHalfValues", Core::DataType::Int64,
                         Core::DataType::Int32, Core::DataType::Half}};
            }

            /// @brief Names a row, which gtest takes as the test suffix.
            std::string dtype_row_name(const ::testing::TestParamInfo<DtypeRow>& info)
            {
                return info.param.name;
            }

            /// @brief Returns a 3 by 4 matrix of five entries with the dtypes of the row.
            SparseMatrix matrix_of(const DtypeRow& row)
            {
                SparseMatrix matrix;
                matrix.rows = 3;
                matrix.cols = 4;
                matrix.row_offsets = vector_of(4, row.offset_dtype);
                matrix.column_indices = vector_of(5, row.column_dtype);
                matrix.values = vector_of(5, row.value_dtype);
                return matrix;
            }
        } // namespace

        /** @brief Builds a matrix with the dtypes of one row at a time. */
        class Small_SparseMatrixDtypes : public ::testing::TestWithParam<DtypeRow>
        {

        };

        TEST_P(Small_SparseMatrixDtypes, ReadsItsCountFromItsValues)
        {
            EXPECT_EQ(matrix_of(GetParam()).nnz(), int64_t{5});
        }

        TEST_P(Small_SparseMatrixDtypes, ReadsItsDtypeFromItsValues)
        {
            EXPECT_EQ(matrix_of(GetParam()).dtype(), GetParam().value_dtype);
        }

        TEST_P(Small_SparseMatrixDtypes, ReadsItsIndexDtypeFromItsRowOffsets)
        {
            EXPECT_EQ(matrix_of(GetParam()).index_dtype(), GetParam().offset_dtype);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_SparseMatrixDtypes, ::testing::ValuesIn(dtype_rows()),
                                 dtype_row_name);
    } // namespace Test
} // namespace SushiBLAS
