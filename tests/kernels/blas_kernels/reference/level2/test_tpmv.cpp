/**************************************************************************/
/* test_tpmv.cpp                                                          */
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
 * @file test_tpmv.cpp
 * @brief Tests the reference tpmv against a dense product with the packed triangle, for every flag.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/tpmv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Names a row after its layout, triangle, transpose and diagonal.
        static std::string row_name(const ::testing::TestParamInfo<std::tuple<bool, bool, bool, bool>>& info)
        {
            return triangular_label(std::get<0>(info.param), std::get<1>(info.param),
                                    std::get<2>(info.param), std::get<3>(info.param));
        }

        /// @brief Runs tpmv for one layout, triangle, transpose and diagonal, as a row of (row major, upper, transposed, unit).
        class Small_ReferenceTpmv : public ::testing::TestWithParam<std::tuple<bool, bool, bool, bool>> {};

        /// @brief Tests that tpmv matches a dense product of the packed triangle for every flag.
        TEST_P(Small_ReferenceTpmv, MatchesADenseProductForEveryFlag)
        {
            const Core::Layout layout = layout_of(std::get<0>(GetParam()));
            const bool upper = std::get<1>(GetParam());
            const bool transposed = std::get<2>(GetParam());
            const bool unit = std::get<3>(GetParam());
            const int64_t n = 5;
            const std::vector<double> a = small_integers<double>(n * n, 1);
            const std::vector<double> start = small_integers<double>(n, 2);
            std::vector<double> x = start;
            const std::vector<double> expected =
                dense_matvec(triangle_of(a, n, upper, unit), n, n, transposed,
                             1.0, start, 0.0, start);
            const std::vector<double> packed =
                to_packed(a, n, uplo_of(upper), layout);

            Kernels::Reference::tpmv(layout, uplo_of(upper),
                                     transpose_of(transposed), diag_of(unit), n,
                                     packed.data(), x.data(), 1);

            EXPECT_EQ(x, expected);
        }

        INSTANTIATE_TEST_SUITE_P(LayoutsTrianglesTransposesAndDiagonals, Small_ReferenceTpmv,
                                 ::testing::Combine(::testing::Bool(), ::testing::Bool(),
                                                    ::testing::Bool(), ::testing::Bool()),
                                 row_name);
    } // namespace Test
} // namespace SushiBLAS
