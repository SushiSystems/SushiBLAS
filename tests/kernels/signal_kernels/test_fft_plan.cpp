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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file test_fft_plan.cpp
 * @brief Tests that FftPlan cooks the radix-2 and Bluestein tables and sizes its scratch.
 * @author Mustafa Garip
 */

#include <SushiBLAS/signal_kernels/fft_plan.hpp>

#include <cmath>
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

        TEST(Small_FftPlan, CooksTheTwiddlesAsUnitRoots)
        {
            const Kernels::FftPlan<double> plan(1, 8, Kernels::FftPairing::ComplexToComplex);
            const Kernels::FftLineView<double> row =
                plan.row_line(plan.values().data(), plan.indices().data());
            const double pi = 3.14159265358979323846;

            for (std::size_t k = 0; k < 4; ++k)
            {
                EXPECT_NEAR(row.twiddles[2 * k], std::cos(-2.0 * pi * k / 8.0), 1e-15) << k;
                EXPECT_NEAR(row.twiddles[2 * k + 1], std::sin(-2.0 * pi * k / 8.0), 1e-15) << k;
            }
        }

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
    } // namespace Test
} // namespace SushiBLAS
