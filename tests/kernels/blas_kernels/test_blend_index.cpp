/**************************************************************************/
/* test_blend_index.cpp                                                   */
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

/**
 * @file test_blend_index.cpp
 * @brief Tests the beta blend of an updated value with its prior value, real and complex, against hand-derived values.
 * @author Mustafa Garip
 */

#include <complex>
#include <limits>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_kernels/blend_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Indexing = Kernels::Indexing;

        namespace
        {
            /// @brief A prior value that fails any comparison it reaches, so a read shows.
            constexpr double K_NAN = std::numeric_limits<double>::quiet_NaN();

            /** @brief Holds a scaled value, a beta, a prior value and the blend they give. */
            struct BlendRow
            {
                double scaled;
                double beta;
                double prior;
                double expected;
            };

            /** @brief Holds the same four for a complex sum and a real beta, as HERK blends. */
            struct ComplexBlendRow
            {
                std::complex<double> scaled;
                double beta;
                std::complex<double> prior;
                std::complex<double> expected;
            };

            /** @brief Checks the blend of a real row. */
            void expect_blend(const BlendRow& row)
            {
                const double got =
                    Indexing::blend_with_prior<double, double>(row.scaled, row.beta, row.prior);
                EXPECT_EQ(got, row.expected);
            }

            /** @brief Checks the blend of a complex row. */
            void expect_complex_blend(const ComplexBlendRow& row)
            {
                using Complex = std::complex<double>;
                const Complex got =
                    Indexing::blend_with_prior<Complex, double>(row.scaled, row.beta, row.prior);
                EXPECT_EQ(got, row.expected);
            }
        } // namespace

        class Small_BlendIndexReal : public ::testing::TestWithParam<BlendRow>
        {
        };

        TEST_P(Small_BlendIndexReal, TheBetaScalesThePriorValueIntoTheSum)
        {
            expect_blend(GetParam());
        }

        // scaled + beta * prior: 3 + 1*5 = 8, 3 + 2*5 = 13, 3 - 5 = -2, 0 + 2*5 = 10; beta 0 gives scaled.
        INSTANTIATE_TEST_SUITE_P(, Small_BlendIndexReal,
                                 ::testing::Values(BlendRow{3.0, 1.0, 5.0, 8.0},
                                                   BlendRow{3.0, 2.0, 5.0, 13.0},
                                                   BlendRow{3.0, -1.0, 5.0, -2.0},
                                                   BlendRow{0.0, 2.0, 5.0, 10.0},
                                                   BlendRow{3.0, 0.0, 5.0, 3.0},
                                                   BlendRow{3.0, 0.0, K_NAN, 3.0}));

        class Small_BlendIndexComplex : public ::testing::TestWithParam<ComplexBlendRow>
        {
        };

        TEST_P(Small_BlendIndexComplex, ARealBetaScalesBothPartsOfThePrior)
        {
            expect_complex_blend(GetParam());
        }

        // (1+2i) + 3*(4+5i) = (1+12) + (2+15)i = 13+17i; a zero beta answers 1+2i.
        INSTANTIATE_TEST_SUITE_P(
            , Small_BlendIndexComplex,
            ::testing::Values(ComplexBlendRow{{1.0, 2.0}, 3.0, {4.0, 5.0}, {13.0, 17.0}},
                              ComplexBlendRow{{1.0, 2.0}, 0.0, {4.0, 5.0}, {1.0, 2.0}}));
    } // namespace Test
} // namespace SushiBLAS
