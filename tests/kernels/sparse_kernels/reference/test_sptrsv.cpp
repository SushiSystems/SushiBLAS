/**************************************************************************/
/* test_sptrsv.cpp                                                        */
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
 * @file test_sptrsv.cpp
 * @brief Tests that the reference sptrsv recovers x from op(A) * x for either triangle.
 * @author Mustafa Garip
 */

#include <string>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/reference/sptrsv.hpp>
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
            /** @brief One flag set of the triangular solve. */
            struct FlagRow
            {
                    const char* name; /**< The row's suffix. */
                    bool upper;       /**< Whether the upper triangle is solved. */
                    bool transposed;  /**< Whether op(A) is the transpose. */
                    bool unit;        /**< Whether the diagonal is taken as one. */
            };

            /** @brief Returns the eight combinations of triangle, transposition and diagonal. */
            std::vector<FlagRow> flag_rows()
            {
                return {{"LowerPlainStored", false, false, false},
                        {"LowerPlainUnit", false, false, true},
                        {"LowerTransposedStored", false, true, false},
                        {"LowerTransposedUnit", false, true, true},
                        {"UpperPlainStored", true, false, false},
                        {"UpperPlainUnit", true, false, true},
                        {"UpperTransposedStored", true, true, false},
                        {"UpperTransposedUnit", true, true, true}};
            }

            /// @brief Names a row, which gtest takes as the test's suffix.
            std::string flag_row_name(const ::testing::TestParamInfo<FlagRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Runs the reference sptrsv one flag set at a time. */
        class Small_ReferenceSptrsv : public ::testing::TestWithParam<FlagRow>
        {

        };

        TEST_P(Small_ReferenceSptrsv, RecoversTheVectorAProductMade)
        {
            const FlagRow& row = GetParam();
            const int64_t n = 9;
            // A stored unit diagonal keeps every substitution exact in integers.
            const IntegerCsr a = integer_csr(n, n, 3, true);
            const Kernels::CsrPattern<int64_t> pattern{n, n, a.nnz(), 0, a.offsets.data(),
                                                       a.columns.data()};
            const std::vector<double> solution = small_integers<double>(n, 4);
            std::vector<double> b = dense_matvec(triangle_of(a.dense(), n, row.upper, row.unit), n,
                                                 n, row.transposed, 1.0, solution, 0.0, solution);

            Kernels::Reference::sptrsv(uplo_of(row.upper), transpose_of(row.transposed),
                                       diag_of(row.unit), pattern, a.values.data(), b.data());

            EXPECT_EQ(b, solution);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_ReferenceSptrsv, ::testing::ValuesIn(flag_rows()),
                                 flag_row_name);
    } // namespace Test
} // namespace SushiBLAS
