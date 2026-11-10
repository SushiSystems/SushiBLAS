/**************************************************************************/
/* test_spmv.cpp                                                          */
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
 * @file test_spmv.cpp
 * @brief Tests the reference spmv against a dense product, plain and transposed.
 * @author Mustafa Garip
 */

#include <limits>
#include <string>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/reference/spmv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include <SushiBLAS/sparse_fixtures/integer_csr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief One call of spmv: the form, the beta, the stride and the starting y. */
            struct SpmvRow
            {
                    const char* name;  /**< The row's suffix. */
                    bool transposed;   /**< Whether the product uses the transpose. */
                    double beta;       /**< The scale of y; zero must not read y. */
                    int64_t inc;       /**< The stride of x and y. */
                    bool poisoned_y;   /**< Whether y starts as NaN, which beta zero must overwrite. */
            };

            /** @brief Returns the rows: both forms with a scaled y, with beta zero over NaN, and with stride 2. */
            std::vector<SpmvRow> spmv_rows()
            {
                return {{"Plain", false, -1.0, 1, false},
                        {"Transposed", true, -1.0, 1, false},
                        {"PlainBetaZeroOverwritesNaN", false, 0.0, 1, true},
                        {"TransposedBetaZeroOverwritesNaN", true, 0.0, 1, true},
                        {"PlainStrided", false, -1.0, 2, false},
                        {"TransposedStrided", true, -1.0, 2, false}};
            }

            /// @brief Names a row, which gtest takes as the test's suffix.
            std::string spmv_row_name(const ::testing::TestParamInfo<SpmvRow>& info)
            {
                return info.param.name;
            }

            /** @brief Spreads a vector over every inc-th slot of a longer one and fills the rest with 99. */
            std::vector<double> spread(const std::vector<double>& dense, int64_t inc)
            {
                std::vector<double> out(dense.size() * static_cast<std::size_t>(inc), 99.0);
                for (std::size_t i = 0; i < dense.size(); ++i)
                    out[i * static_cast<std::size_t>(inc)] = dense[i];
                return out;
            }

            /** @brief Returns every inc-th slot of a spread vector. */
            std::vector<double> gather(const std::vector<double>& spread_out, int64_t inc)
            {
                std::vector<double> out;
                for (std::size_t i = 0; i < spread_out.size(); i += static_cast<std::size_t>(inc))
                    out.push_back(spread_out[i]);
                return out;
            }
        } // namespace

        /** @brief Runs the reference spmv one row at a time. */
        class Small_ReferenceCsrSpmv : public ::testing::TestWithParam<SpmvRow>
        {

        };

        TEST_P(Small_ReferenceCsrSpmv, MatchesADenseProduct)
        {
            const SpmvRow& row = GetParam();
            const IntegerCsr a = integer_csr(9, 7, 1, false);
            const Kernels::CsrPattern<int64_t> pattern{a.rows, a.cols, a.nnz(), 0,
                                                       a.offsets.data(), a.columns.data()};
            const std::vector<double> x = small_integers<double>(row.transposed ? 9 : 7, 2);
            const std::vector<double> y_start = small_integers<double>(row.transposed ? 7 : 9, 3);
            const std::vector<double> expected_start =
                row.beta == 0.0 ? std::vector<double>(y_start.size(), 0.0) : y_start;
            const std::vector<double> expected =
                dense_matvec(a.dense(), 9, 7, row.transposed, 2.0, x, row.beta, expected_start);
            const std::vector<double> poison(y_start.size(),
                                             std::numeric_limits<double>::quiet_NaN());
            const std::vector<double> x_stored = spread(x, row.inc);
            std::vector<double> y_stored = spread(row.poisoned_y ? poison : y_start, row.inc);

            Kernels::Reference::spmv(transpose_of(row.transposed), pattern, a.values.data(), 2.0,
                                     x_stored.data(), row.inc, row.beta, y_stored.data(), row.inc);

            EXPECT_EQ(gather(y_stored, row.inc), expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_ReferenceCsrSpmv, ::testing::ValuesIn(spmv_rows()),
                                 spmv_row_name);
    } // namespace Test
} // namespace SushiBLAS
