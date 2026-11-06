/**************************************************************************/
/* test_spmm.cpp                                                          */
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
 * @file test_spmm.cpp
 * @brief Tests the reference spmm column by column against a dense product, in both layouts and
 * both forms.
 * @author Mustafa Garip
 */

#include <string>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/reference/spmm.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include <SushiBLAS/sparse_fixtures/integer_csr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief One layout and one form of the spmm. */
            struct SpmmRow
            {
                    const char* name;    /**< The row's suffix. */
                    Core::Layout layout; /**< The layout of B and C. */
                    bool transposed;     /**< Whether op(A) is the transpose. */
            };

            /** @brief Returns the four combinations of layout and form. */
            std::vector<SpmmRow> spmm_rows()
            {
                return {{"RowMajorPlain", Core::Layout::RowMajor, false},
                        {"RowMajorTransposed", Core::Layout::RowMajor, true},
                        {"ColumnMajorPlain", Core::Layout::ColumnMajor, false},
                        {"ColumnMajorTransposed", Core::Layout::ColumnMajor, true}};
            }

            /// @brief Names a row, which gtest takes as the test's suffix.
            std::string spmm_row_name(const ::testing::TestParamInfo<SpmmRow>& info)
            {
                return info.param.name;
            }

            /** @brief Returns column j of a row-major dense matrix of rows rows and nrhs columns. */
            std::vector<double> column_of(const std::vector<double>& dense, int64_t rows,
                                          int64_t nrhs, int64_t j)
            {
                std::vector<double> out(static_cast<std::size_t>(rows));
                for (int64_t i = 0; i < rows; ++i)
                    out[static_cast<std::size_t>(i)] = element(dense, nrhs, i, j);
                return out;
            }

            /** @brief Returns 2 op(A) B + C computed one column at a time with the dense oracle, row-major. */
            std::vector<double> dense_product(const IntegerCsr& a, bool transposed,
                                              const std::vector<double>& b,
                                              const std::vector<double>& c, int64_t nrhs)
            {
                const int64_t in_rows = transposed ? a.rows : a.cols;
                const int64_t out_rows = transposed ? a.cols : a.rows;
                std::vector<double> expected(static_cast<std::size_t>(out_rows * nrhs));
                for (int64_t j = 0; j < nrhs; ++j)
                {
                    const std::vector<double> product = dense_matvec(
                        a.dense(), a.rows, a.cols, transposed, 2.0, column_of(b, in_rows, nrhs, j),
                        1.0, column_of(c, out_rows, nrhs, j));
                    for (int64_t i = 0; i < out_rows; ++i)
                        expected[static_cast<std::size_t>(i * nrhs + j)] =
                            product[static_cast<std::size_t>(i)];
                }
                return expected;
            }
        } // namespace

        /** @brief Runs the reference spmm one layout and form at a time. */
        class Small_ReferenceSpmm : public ::testing::TestWithParam<SpmmRow>
        {

        };

        TEST_P(Small_ReferenceSpmm, EachColumnOfCIsAProductOfTheMatrixWithAColumnOfB)
        {
            const SpmmRow& row = GetParam();
            const int64_t rows = 7;
            const int64_t cols = 6;
            const int64_t nrhs = 3;
            const IntegerCsr a = integer_csr(rows, cols, 2, false);
            const Kernels::CsrPattern<int64_t> pattern{rows, cols, a.nnz(), 0, a.offsets.data(),
                                                       a.columns.data()};
            const int64_t in_rows = row.transposed ? rows : cols;
            const int64_t out_rows = row.transposed ? cols : rows;
            const std::vector<double> b = small_integers<double>(in_rows * nrhs, 3);
            const std::vector<double> c = small_integers<double>(out_rows * nrhs, 4);
            const std::vector<double> expected = dense_product(a, row.transposed, b, c, nrhs);
            const bool row_major = (row.layout == Core::Layout::RowMajor);
            const std::vector<double> stored_b = in_layout(b, in_rows, nrhs, row.layout);
            std::vector<double> stored_c = in_layout(c, out_rows, nrhs, row.layout);

            Kernels::Reference::spmm(row.layout, transpose_of(row.transposed), pattern,
                                     a.values.data(), nrhs, 2.0, stored_b.data(),
                                     row_major ? nrhs : in_rows, 1.0, stored_c.data(),
                                     row_major ? nrhs : out_rows);

            EXPECT_EQ(stored_c, in_layout(expected, out_rows, nrhs, row.layout));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_ReferenceSpmm, ::testing::ValuesIn(spmm_rows()),
                                 spmm_row_name);
    } // namespace Test
} // namespace SushiBLAS
