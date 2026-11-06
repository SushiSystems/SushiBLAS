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
 * @file test_convolve_1d.cpp
 * @brief Tests the reference valid-mode convolution against hand values and a direct loop.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/signal_kernels/reference/convolve_1d.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
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

        namespace
        {
            /// @brief Holds the rows of a signal, its taps and the convolution they give.
            struct ConvolutionRow
            {
                    const char* name;
                    int64_t rows;
                    int64_t length;
                    std::vector<double> signal;
                    std::vector<double> taps;
                    std::vector<double> expected;
            };

            /**
             * @brief Returns the valid-mode convolution one output at a time.
             *
             * Output n of a row is the sum over j of signal[n + j] * taps[taps - 1 - j].
             */
            std::vector<double> direct_convolution(int64_t rows, int64_t length,
                                                   const std::vector<double>& signal,
                                                   const std::vector<double>& taps)
            {
                const int64_t tap_count = static_cast<int64_t>(taps.size());
                const int64_t out_length = length - tap_count + 1;
                std::vector<double> expected(static_cast<std::size_t>(rows * out_length), 0.0);
                for (int64_t row = 0; row < rows; ++row)
                    for (int64_t n = 0; n < out_length; ++n)
                        for (int64_t j = 0; j < tap_count; ++j)
                            expected[static_cast<std::size_t>(row * out_length + n)] +=
                                signal[static_cast<std::size_t>(row * length + n + j)]
                                * taps[static_cast<std::size_t>(tap_count - 1 - j)];
                return expected;
            }

            /// @brief Returns a hand-worked row of reversed taps and a row of small integers.
            std::vector<ConvolutionRow> convolution_rows()
            {
                const std::vector<double> integer_signal = small_integers<double>(3 * 11, 1);
                const std::vector<double> integer_taps = small_integers<double>(4, 2);
                return {{"TapsAppliedInReverse", 1, 5, {1, 2, 3, 4, 5}, {10, 1}, {21, 32, 43, 54}},
                        {"SmallIntegers", 3, 11, integer_signal, integer_taps,
                         direct_convolution(3, 11, integer_signal, integer_taps)}};
            }

            /// @brief Names a row, which gtest takes as the test's suffix.
            std::string convolution_row_name(const ::testing::TestParamInfo<ConvolutionRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /// @brief Convolves the signal of one row.
        class Small_ReferenceConvolve1DRows : public ::testing::TestWithParam<ConvolutionRow>
        {
        };

        TEST_P(Small_ReferenceConvolve1DRows, MatchesADirectLoopOnSmallIntegers)
        {
            const ConvolutionRow& row = GetParam();
            std::vector<double> out(row.expected.size(), 0.0);

            Kernels::Reference::convolve_1d(row.rows, row.length,
                                            static_cast<int64_t>(row.taps.size()),
                                            row.signal.data(), row.taps.data(), out.data());

            EXPECT_EQ(out, row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_ReferenceConvolve1DRows, ::testing::ValuesIn(convolution_rows()),
                                 convolution_row_name);
    } // namespace Test
} // namespace SushiBLAS
