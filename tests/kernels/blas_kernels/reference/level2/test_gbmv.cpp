/**************************************************************************/
/* test_gbmv.cpp                                                          */
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
 * @file test_gbmv.cpp
 * @brief Tests the reference gbmv against a loop over the band, in both layouts and transposes.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/gbmv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Names a row after its layout and its transpose flag.
        static std::string row_name(const ::testing::TestParamInfo<std::tuple<bool, bool>>& info)
        {
            return layout_transpose_label(std::get<0>(info.param), std::get<1>(info.param));
        }

        /// @brief Runs gbmv for one layout and one transpose flag, as a row of (row major, transposed).
        class Small_ReferenceGbmv : public ::testing::TestWithParam<std::tuple<bool, bool>> {};

        /// @brief Tests that gbmv matches a dense product that reads only the band.
        TEST_P(Small_ReferenceGbmv, MatchesADenseLoopOverTheBand)
        {
            const Core::Layout layout = layout_of(std::get<0>(GetParam()));
            const bool transposed = std::get<1>(GetParam());
            const int64_t m = 5;
            const int64_t n = 6;
            const int64_t kl = 1;
            const int64_t ku = 2;
            const int64_t ld = kl + ku + 2;
            const std::vector<double> dense = band_of(small_integers<double>(m * n, 1), m, n,
                                                      kl, ku);
            const std::vector<double> x = small_integers<double>(transposed ? m : n, 2);
            std::vector<double> y = small_integers<double>(transposed ? n : m, 3);
            const std::vector<double> expected = dense_matvec(dense, m, n, transposed, 2.0, x, 1.0, y);
            const std::vector<double> band = to_band(dense, m, n, kl, ku, ld, layout, 99.0);

            Kernels::Reference::gbmv(layout, transpose_of(transposed), m, n, kl, ku, 2.0,
                                     band.data(), ld, x.data(), 1, 1.0, y.data(), 1);

            EXPECT_EQ(y, expected);
        }

        INSTANTIATE_TEST_SUITE_P(LayoutsAndTransposes, Small_ReferenceGbmv,
                                 ::testing::Combine(::testing::Bool(), ::testing::Bool()), row_name);
    } // namespace Test
} // namespace SushiBLAS
