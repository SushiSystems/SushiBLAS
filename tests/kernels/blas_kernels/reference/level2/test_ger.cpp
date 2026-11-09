/**************************************************************************/
/* test_ger.cpp                                                           */
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
 * @file test_ger.cpp
 * @brief Tests the reference ger, A += alpha * x * y^T, in both layouts and with a conjugated y.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/ger.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns a, row by row, with alpha * x * y^T added. */
            std::vector<double> with_outer_product(std::vector<double> a, int64_t n, double alpha,
                                                   const std::vector<double>& x,
                                                   const std::vector<double>& y)
            {
                for (std::size_t i = 0; i < x.size(); ++i)
                {
                    for (std::size_t j = 0; j < y.size(); ++j)
                    {
                        a[i * static_cast<std::size_t>(n) + j] += alpha * x[i] * y[j];
                    }
                }

                return a;
            }

            /** @brief Names a row after its layout. */
            std::string layout_name(const ::testing::TestParamInfo<bool>& info)
            {
                return info.param ? "RowMajor" : "ColumnMajor";
            }
        } // namespace

        /// @brief Runs ger for one layout, as a row of (row major).
        class Small_ReferenceGerLayout : public ::testing::TestWithParam<bool> {};

        /// @brief Tests that ger adds alpha x y^T to A in the layout of the row.
        TEST_P(Small_ReferenceGerLayout, AddsTheOuterProductInEitherLayout)
        {
            const Core::Layout layout = layout_of(GetParam());
            const int64_t m = 3;
            const int64_t n = 4;
            const std::vector<double> a = small_integers<double>(m * n, 1);
            const std::vector<double> x = small_integers<double>(m, 2);
            const std::vector<double> y = small_integers<double>(n, 3);
            const std::vector<double> expected = with_outer_product(a, n, -2.0, x, y);
            std::vector<double> stored = in_layout(a, m, n, layout);
            const int64_t lda = (layout == Core::Layout::RowMajor) ? n : m;

            Kernels::Reference::ger(layout, m, n, -2.0, x.data(), 1, y.data(), 1, stored.data(), lda);

            EXPECT_EQ(stored, in_layout(expected, m, n, layout));
        }

        INSTANTIATE_TEST_SUITE_P(Layouts, Small_ReferenceGerLayout, ::testing::Bool(), layout_name);

        TEST(Small_ReferenceGer, TheConjugatedFormConjugatesY)
        {
            const std::vector<std::complex<double>> x{{1.0, 1.0}};
            const std::vector<std::complex<double>> y{{0.0, 2.0}};
            std::vector<std::complex<double>> a{{0.0, 0.0}};

            Kernels::Reference::ger<std::complex<double>, Kernels::Conjugated>(
                Core::Layout::RowMajor, 1, 1, std::complex<double>(1.0, 0.0), x.data(), 1,
                y.data(), 1, a.data(), 1);

            EXPECT_EQ(a[0], std::complex<double>(2.0, -2.0));
        }
    } // namespace Test
} // namespace SushiBLAS
