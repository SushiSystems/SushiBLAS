/**************************************************************************/
/* test_spsymv.cpp                                                        */
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
 * @file test_spsymv.cpp
 * @brief Tests the reference spsymv against a dense product with the mirrored triangle.
 * @author Mustafa Garip
 */

#include <string>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/reference/spsymv.hpp>
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
            /** @brief The triangle spsymv reads. */
            struct TriangleRow
            {
                    const char* name; /**< The row's suffix. */
                    bool upper;       /**< Whether the upper triangle is the one stored and read. */
            };

            /// @brief Names a row, which gtest takes as the test's suffix.
            std::string triangle_row_name(const ::testing::TestParamInfo<TriangleRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Runs the reference spsymv one triangle at a time. */
        class Small_ReferenceSpsymv : public ::testing::TestWithParam<TriangleRow>
        {

        };

        TEST_P(Small_ReferenceSpsymv, ReadsOneTriangleAndSkipsTheOther)
        {
            const bool upper = GetParam().upper;
            const int64_t n = 8;
            const IntegerCsr a = integer_csr(n, n, 4, true);
            const Kernels::CsrPattern<int64_t> pattern{n, n, a.nnz(), 0, a.offsets.data(),
                                                       a.columns.data()};
            const std::vector<double> x = small_integers<double>(n, 5);
            std::vector<double> y = small_integers<double>(n, 6);
            const std::vector<double> expected =
                dense_matvec(symmetric_from(a.dense(), n, upper), n, n, false, 1.0, x, 2.0, y);

            Kernels::Reference::spsymv(uplo_of(upper), pattern, a.values.data(), 1.0, x.data(), 1,
                                       2.0, y.data(), 1);

            EXPECT_EQ(y, expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_ReferenceSpsymv,
                                 ::testing::Values(TriangleRow{"Lower", false},
                                                   TriangleRow{"Upper", true}),
                                 triangle_row_name);
    } // namespace Test
} // namespace SushiBLAS
