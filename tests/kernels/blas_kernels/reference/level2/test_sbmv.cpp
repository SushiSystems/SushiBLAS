/**************************************************************************/
/* test_sbmv.cpp                                                          */
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
 * @file test_sbmv.cpp
 * @brief Tests the reference sbmv against a dense loop over the mirrored band triangle.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/sbmv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Names a row after its layout and its triangle.
        static std::string row_name(const ::testing::TestParamInfo<std::tuple<bool, bool>>& info)
        {
            return layout_uplo_label(std::get<0>(info.param), std::get<1>(info.param));
        }

        /// @brief Runs sbmv for one layout and one triangle, as a row of (row major, upper).
        class Small_ReferenceSbmv : public ::testing::TestWithParam<std::tuple<bool, bool>> {};

        /// @brief Tests that sbmv matches a dense product that reads only the named band triangle.
        TEST_P(Small_ReferenceSbmv, MatchesADenseLoopOverTheBandTriangle)
        {
            const Core::Layout layout = layout_of(std::get<0>(GetParam()));
            const bool upper = std::get<1>(GetParam());
            const int64_t n = 6;
            const int64_t k = 2;
            const int64_t ld = k + 2;
            const std::vector<double> a = small_integers<double>(n * n, 1);
            const std::vector<double> symmetric =
                band_of(symmetric_from(a, n, upper), n, n, k, k);
            const std::vector<double> x = small_integers<double>(n, 2);
            std::vector<double> y = small_integers<double>(n, 3);
            const std::vector<double> expected =
                dense_matvec(symmetric, n, n, false, -1.0, x, 2.0, y);
            const std::vector<double> band =
                to_triangle_band(symmetric, n, uplo_of(upper), k, ld, layout, 99.0);

            Kernels::Reference::sbmv(layout, uplo_of(upper), n, k, -1.0, band.data(), ld,
                                     x.data(), 1, 2.0, y.data(), 1);

            EXPECT_EQ(y, expected);
        }

        INSTANTIATE_TEST_SUITE_P(LayoutsAndTriangles, Small_ReferenceSbmv,
                                 ::testing::Combine(::testing::Bool(), ::testing::Bool()), row_name);
    } // namespace Test
} // namespace SushiBLAS
