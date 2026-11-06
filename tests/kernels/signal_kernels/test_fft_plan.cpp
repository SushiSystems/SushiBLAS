/**************************************************************************/
/* test_fft_plan.cpp                                                      */
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
 * @file test_fft_plan.cpp
 * @brief Tests that FftPlan cooks the radix-2 and Bluestein tables and sizes its scratch.
 * @author Mustafa Garip
 */

#include <SushiBLAS/signal_kernels/fft_plan.hpp>
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_FftPlan, CooksTheBitReversalOfAPowerOfTwoLine)
        {
            const Kernels::FftPlan<double> plan(1, 8, Kernels::FftPairing::ComplexToComplex);
            const Kernels::FftLineView<double> row =
                plan.row_line(plan.values().data(), plan.indices().data());

            ASSERT_EQ(row.length, 8u);
            ASSERT_EQ(row.power_size, 8u);
            EXPECT_FALSE(row.is_bluestein());
            const std::vector<std::uint32_t> expected{0, 4, 2, 6, 1, 5, 3, 7};
            EXPECT_EQ(std::vector<std::uint32_t>(row.reversal, row.reversal + 8), expected);
        }

        /// @brief The absolute tolerance of a twiddle: one rounding of a value of order one.
        constexpr double K_TWIDDLE_TOLERANCE = absolute_tolerance<double>(1);

        /// @brief One twiddle of a line of eight and the unit root exp(-2 pi i k / 8) worked by hand.
        struct TwiddleRow
        {
                const char* name;
                std::size_t k;
                double real;
                double imaginary;
        };

        /// @brief Returns the four roots a line of eight keeps; sqrt(2) / 2 is 0.7071067811865476.
        static std::vector<TwiddleRow> twiddle_rows()
        {
            return {{"K0", 0, 1.0, 0.0},
                    {"K1", 1, 0.7071067811865476, -0.7071067811865476},
                    {"K2", 2, 0.0, -1.0},
                    {"K3", 3, -0.7071067811865476, -0.7071067811865476}};
        }

        /// @brief Names a row, which gtest takes as the test suffix.
        static std::string twiddle_row_name(const ::testing::TestParamInfo<TwiddleRow>& info)
        {
            return info.param.name;
        }

        /// @brief Reads one twiddle of a plan of eight.
        class Small_FftPlanTwiddles : public ::testing::TestWithParam<TwiddleRow>
        {
        };

        TEST_P(Small_FftPlanTwiddles, CooksTheTwiddlesAsUnitRoots)
        {
            const TwiddleRow& row = GetParam();
            const Kernels::FftPlan<double> plan(1, 8, Kernels::FftPairing::ComplexToComplex);
            const Kernels::FftLineView<double> line =
                plan.row_line(plan.values().data(), plan.indices().data());

            EXPECT_NEAR(line.twiddles[2 * row.k], row.real, K_TWIDDLE_TOLERANCE);
            EXPECT_NEAR(line.twiddles[2 * row.k + 1], row.imaginary, K_TWIDDLE_TOLERANCE);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_FftPlanTwiddles, ::testing::ValuesIn(twiddle_rows()),
                                 twiddle_row_name);

        TEST(Small_FftPlan, PadsANonPowerOfTwoLineForBluestein)
        {
            const Kernels::FftPlan<float> plan(1, 6, Kernels::FftPairing::ComplexToComplex);
            const Kernels::FftLineView<float> row =
                plan.row_line(plan.values().data(), plan.indices().data());

            EXPECT_EQ(row.length, 6u);
            // The padding is the least power of two at or above 2 * 6 - 1.
            EXPECT_EQ(row.power_size, 16u);
            EXPECT_TRUE(row.is_bluestein());
            EXPECT_EQ(plan.padded_scalars(3), 2u * 3u * 16u);
        }

        TEST(Small_FftPlan, PacksAnEvenRealRowIntoAHalfLine)
        {
            const Kernels::FftPlan<double> even(2, 8, Kernels::FftPairing::RealToComplex);
            const Kernels::FftPlan<double> odd(2, 7, Kernels::FftPairing::RealToComplex);

            EXPECT_TRUE(even.is_packed());
            EXPECT_EQ(even.row_line(even.values().data(), even.indices().data()).length, 4u);
            EXPECT_EQ(even.spectrum_columns(), 5u);
            EXPECT_NE(even.half_roots(even.values().data()), nullptr);
            EXPECT_FALSE(odd.is_packed());
            EXPECT_EQ(odd.row_line(odd.values().data(), odd.indices().data()).length, 7u);
            EXPECT_EQ(odd.spectrum_columns(), 4u);
            EXPECT_EQ(odd.half_roots(odd.values().data()), nullptr);
        }

        TEST(Small_FftPlan, CountsTheScalarsOfEachSide)
        {
            const Kernels::FftPlan<double> complex(3, 5, Kernels::FftPairing::ComplexToComplex);
            const Kernels::FftPlan<double> real(3, 6, Kernels::FftPairing::RealToComplex);

            EXPECT_EQ(complex.sample_scalars(), 30u);
            EXPECT_EQ(complex.spectrum_scalars(), 30u);
            EXPECT_EQ(complex.line_scalars(2), 0u);
            EXPECT_EQ(real.sample_scalars(), 18u);
            EXPECT_EQ(real.spectrum_scalars(), 24u);
            EXPECT_EQ(real.line_scalars(2), 2u * 2u * 3u * 3u);
        }

        TEST(Small_FftPlan, GivesOneRowNoColumnLine)
        {
            const Kernels::FftPlan<float> plan(1, 4, Kernels::FftPairing::ComplexToComplex);

            EXPECT_EQ(plan.column_line(plan.values().data(), plan.indices().data()).length, 0u);
        }

        TEST(Small_FftPlan, GivesAMultiRowPlanAColumnLineAsLongAsItsRows)
        {
            const Kernels::FftPlan<float> plan(3, 4, Kernels::FftPairing::ComplexToComplex);

            EXPECT_EQ(plan.column_line(plan.values().data(), plan.indices().data()).length, 3u);
        }
    } // namespace Test
} // namespace SushiBLAS
