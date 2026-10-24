/**************************************************************************/
/* test_convolve_1d.cpp                                                   */
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

/**
 * @file test_convolve_1d.cpp
 * @brief Tests the reference valid-mode convolution against hand values and a direct loop.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/signal_kernels/reference/convolve_1d.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_ReferenceConvolve1D, AppliesTheTapsInReverseOverTheSignal)
        {
            const std::vector<float> signal{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            const std::vector<float> taps{10.0f, 1.0f};
            std::vector<float> out(4, 0.0f);

            Kernels::Reference::convolve_1d(1, 5, 2, signal.data(), taps.data(), out.data());

            EXPECT_EQ(out, (std::vector<float>{21, 32, 43, 54}));
        }

        TEST(Small_ReferenceConvolve1D, ConvolvesEachRowOnItsOwn)
        {
            const std::vector<double> signal{1, 2, 3, 100, 200, 300};
            const std::vector<double> taps{1, 1};
            std::vector<double> out(4, 0.0);

            Kernels::Reference::convolve_1d(2, 3, 2, signal.data(), taps.data(), out.data());

            EXPECT_EQ(out, (std::vector<double>{3, 5, 300, 500}));
        }

        TEST(Small_ReferenceConvolve1D, OneTapScalesTheRow)
        {
            const std::vector<double> signal{1, 2, 3, 4};
            const std::vector<double> taps{3};
            std::vector<double> out(4, 0.0);

            Kernels::Reference::convolve_1d(2, 2, 1, signal.data(), taps.data(), out.data());

            EXPECT_EQ(out, (std::vector<double>{3, 6, 9, 12}));
        }

        TEST(Small_ReferenceConvolve1D, TapsAsLongAsTheRowGiveOneSamplePerRow)
        {
            const std::vector<double> signal{1, 2, 3, 4, 5, 6};
            const std::vector<double> taps{1, 10, 100};
            std::vector<double> out(2, 0.0);

            Kernels::Reference::convolve_1d(2, 3, 3, signal.data(), taps.data(), out.data());

            // Each row gives signal[2] * 1 + signal[1] * 10 + signal[0] * 100.
            EXPECT_EQ(out, (std::vector<double>{123, 456}));
        }

        TEST(Small_ReferenceConvolve1D, MatchesADirectLoopOnSmallIntegers)
        {
            constexpr int64_t K_ROWS = 3;
            constexpr int64_t K_LENGTH = 11;
            constexpr int64_t K_TAPS = 4;
            constexpr int64_t K_OUT_LENGTH = K_LENGTH - K_TAPS + 1;
            const std::vector<double> signal = small_integers<double>(K_ROWS * K_LENGTH, 1);
            const std::vector<double> taps = small_integers<double>(K_TAPS, 2);
            std::vector<double> out(K_ROWS * K_OUT_LENGTH, 0.0);
            std::vector<double> expected(K_ROWS * K_OUT_LENGTH, 0.0);

            Kernels::Reference::convolve_1d(K_ROWS, K_LENGTH, K_TAPS, signal.data(), taps.data(),
                                            out.data());

            for (int64_t row = 0; row < K_ROWS; ++row)
                for (int64_t n = 0; n < K_OUT_LENGTH; ++n)
                    for (int64_t j = 0; j < K_TAPS; ++j)
                        expected[static_cast<std::size_t>(row * K_OUT_LENGTH + n)] +=
                            signal[static_cast<std::size_t>(row * K_LENGTH + n + j)]
                            * taps[static_cast<std::size_t>(K_TAPS - 1 - j)];

            EXPECT_EQ(out, expected);
        }
    } // namespace Test
} // namespace SushiBLAS
