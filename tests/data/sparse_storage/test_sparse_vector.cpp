/**************************************************************************/
/* test_sparse_vector.cpp                                                 */
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
 * @file test_sparse_vector.cpp
 * @brief Tests that a SparseVector reads its entry count and both dtypes from its tensors.
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
#include <SushiBLAS/sparse_storage/sparse_vector.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Returns a rank-one tensor of count elements of dtype, with no storage.
        static Tensor entries_of(int64_t count, Core::DataType dtype)
        {
            const int64_t dims[1] = {count};
            return Tensor::make_shape_only(SushiRuntime::span<const int64_t>(dims, 1), dtype);
        }

        TEST(Small_SparseVector, ADefaultVectorIsEmpty)
        {
            const SparseVector vector;

            EXPECT_EQ(vector.size, int64_t{0});
            EXPECT_EQ(vector.nnz(), int64_t{0});
        }

        TEST(Small_SparseVector, ADefaultVectorIsZeroBased)
        {
            const SparseVector vector;

            EXPECT_EQ(vector.index_base, Core::IndexBase::Zero);
        }

        namespace
        {
            /** @brief Holds the dtypes of the two arrays of one vector. */
            struct DtypeRow
            {
                const char* name;               /**< The row suffix. */
                Core::DataType index_dtype;     /**< The element type of the indices. */
                Core::DataType value_dtype;     /**< The element type of the values. */
            };

            /** @brief Returns the dtype pairs the vector is read with. */
            std::vector<DtypeRow> dtype_rows()
            {
                return {{"Int64IndicesHalfValues", Core::DataType::Int64, Core::DataType::Half},
                        {"Int32IndicesComplex64Values", Core::DataType::Int32,
                         Core::DataType::Complex64}};
            }

            /// @brief Names a row, which gtest takes as the test suffix.
            std::string dtype_row_name(const ::testing::TestParamInfo<DtypeRow>& info)
            {
                return info.param.name;
            }

            /// @brief Returns a vector of three entries with the dtypes of the row.
            SparseVector vector_of(const DtypeRow& row)
            {
                SparseVector vector;
                vector.size = 10;
                vector.index_base = Core::IndexBase::One;
                vector.indices = entries_of(3, row.index_dtype);
                vector.values = entries_of(3, row.value_dtype);
                return vector;
            }
        } // namespace

        /** @brief Builds a vector with the dtypes of one row at a time. */
        class Small_SparseVectorDtypes : public ::testing::TestWithParam<DtypeRow>
        {

        };

        TEST_P(Small_SparseVectorDtypes, ReadsItsCountFromItsValues)
        {
            EXPECT_EQ(vector_of(GetParam()).nnz(), int64_t{3});
        }

        TEST_P(Small_SparseVectorDtypes, ReadsItsDtypeFromItsValues)
        {
            EXPECT_EQ(vector_of(GetParam()).dtype(), GetParam().value_dtype);
        }

        TEST_P(Small_SparseVectorDtypes, ReadsItsIndexDtypeFromItsIndices)
        {
            EXPECT_EQ(vector_of(GetParam()).index_dtype(), GetParam().index_dtype);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_SparseVectorDtypes, ::testing::ValuesIn(dtype_rows()),
                                 dtype_row_name);
    } // namespace Test
} // namespace SushiBLAS
