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
 * @brief Tests the reference spmv against a dense loop over the mirrored packed triangle.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/spmv.hpp>
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

        /// @brief Runs spmv for one layout and one triangle, as a row of (row major, upper).
        class Small_ReferenceSpmv : public ::testing::TestWithParam<std::tuple<bool, bool>> {};

        /// @brief Tests that spmv matches a dense product that reads only the named packed triangle.
        TEST_P(Small_ReferenceSpmv, MatchesADenseLoopOverThePackedTriangle)
        {
            const Core::Layout layout = layout_of(std::get<0>(GetParam()));
            const bool upper = std::get<1>(GetParam());
            const int64_t n = 5;
            const std::vector<double> a = small_integers<double>(n * n, 1);
            const std::vector<double> x = small_integers<double>(n, 2);
            std::vector<double> y = small_integers<double>(n, 3);
            const std::vector<double> expected =
                dense_matvec(symmetric_from(a, n, upper), n, n, false, 2.0, x, 0.0, y);
            const std::vector<double> packed = to_packed(a, n, uplo_of(upper), layout);

            Kernels::Reference::spmv(layout, uplo_of(upper), n, 2.0, packed.data(),
                                     x.data(), 1, 0.0, y.data(), 1);

            EXPECT_EQ(y, expected);
        }

        INSTANTIATE_TEST_SUITE_P(LayoutsAndTriangles, Small_ReferenceSpmv,
                                 ::testing::Combine(::testing::Bool(), ::testing::Bool()), row_name);
    } // namespace Test
} // namespace SushiBLAS
