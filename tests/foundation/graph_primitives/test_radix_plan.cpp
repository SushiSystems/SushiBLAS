/**************************************************************************/
/* test_radix_plan.cpp                                                    */
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
 * @file test_radix_plan.cpp
 * @brief Tests the digit count, the digit of a key and the per-digit plan, without a device.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <SushiBLAS/graph_primitives/radix_plan.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using GraphPrimitives::Detail::fill_radix_plan;
            using GraphPrimitives::Detail::radix_digit;
            using GraphPrimitives::Detail::radix_digit_count;
            namespace Plan = GraphPrimitives::Detail;

            /// @brief Names a key range and the digits it needs.
            struct DigitCountRow
            {
                /// @brief Holds the key range.
                std::size_t key_range;
                /// @brief Holds the number of 8-bit digits the range needs.
                std::size_t digits;
            };

            /// @brief Names a shift and the byte of the key at it.
            struct DigitRow
            {
                /// @brief Holds the shift in bits.
                std::uint32_t shift;
                /// @brief Holds the byte of the key at the shift.
                std::uint32_t digit;
            };

            /// @brief Names a digit of a plan, how many digits are live and what the digit does.
            struct PlanRow
            {
                /// @brief Holds the number of digits the live key range needs.
                std::size_t live_digits;
                /// @brief Holds the digit whose words are read.
                std::size_t digit;
                /// @brief Holds whether the digit runs.
                bool active;
                /// @brief Holds the word telling whether the digit writes the indices.
                std::uint32_t into_indices;
            };
        } // namespace

        /// @brief Runs one key range of the digit count.
        class Small_GraphPrimitivesRadixDigitCount : public ::testing::TestWithParam<DigitCountRow>
        {
        };

        /** @brief Tests the digit count at the 8-bit boundaries of the key range. */
        TEST_P(Small_GraphPrimitivesRadixDigitCount, DigitCountAtByteBoundaries)
        {
            EXPECT_EQ(radix_digit_count(GetParam().key_range), GetParam().digits);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_GraphPrimitivesRadixDigitCount,
            ::testing::Values(DigitCountRow{0, 1u}, DigitCountRow{1, 1u},
                              DigitCountRow{std::size_t{1} << 8, 1u},
                              DigitCountRow{(std::size_t{1} << 8) + 1, 2u},
                              DigitCountRow{std::size_t{1} << 16, 2u},
                              DigitCountRow{(std::size_t{1} << 16) + 1, 3u},
                              DigitCountRow{std::size_t{1} << 24, 3u},
                              DigitCountRow{(std::size_t{1} << 24) + 1, 4u}));

        /// @brief Runs one shift of the digit of a key.
        class Small_GraphPrimitivesRadixDigit : public ::testing::TestWithParam<DigitRow>
        {
        };

        /** @brief Tests that the digit of a key is the byte at its shift. */
        TEST_P(Small_GraphPrimitivesRadixDigit, DigitIsTheByteAtItsShift)
        {
            constexpr std::uint32_t K_KEY = 0xA1B2C3D4u;

            EXPECT_EQ(radix_digit(K_KEY, GetParam().shift), GetParam().digit);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesRadixDigit,
                                 ::testing::Values(DigitRow{0, 0xD4u}, DigitRow{8, 0xC3u},
                                                   DigitRow{16, 0xB2u}, DigitRow{24, 0xA1u}));

        /// @brief Runs one digit of a plan built for a live key range.
        class Small_GraphPrimitivesRadixPlanDigit : public ::testing::TestWithParam<PlanRow>
        {
        };

        /** @brief Tests that a three-digit plan runs only the digits the live key range needs. */
        TEST_P(Small_GraphPrimitivesRadixPlanDigit, PlanSkipsDigitsAboveTheLiveKeyRange)
        {
            constexpr std::size_t K_DIGITS = 3;
            constexpr std::size_t K_LIVE_COUNT = 1000;
            constexpr std::size_t K_LIVE_TILES = 4;
            const PlanRow& row = GetParam();
            std::vector<std::uint32_t> plan(K_DIGITS * Plan::K_RADIX_PLAN_WORDS, 0xFFFFFFFFu);

            fill_radix_plan(K_DIGITS, K_LIVE_COUNT, K_LIVE_TILES,
                            std::size_t{1} << (8 * row.live_digits), plan.data());

            const auto word = [&plan, &row](std::size_t offset)
            { return plan[row.digit * Plan::K_RADIX_PLAN_WORDS + offset]; };
            EXPECT_EQ(word(Plan::K_RADIX_PLAN_TILES), row.active ? K_LIVE_TILES : 0u);
            EXPECT_EQ(word(Plan::K_RADIX_PLAN_BINS), row.active ? Plan::K_RADIX_DIGIT_BINS : 0u);
            EXPECT_EQ(word(Plan::K_RADIX_PLAN_ONE), row.active ? 1u : 0u);
            EXPECT_EQ(word(Plan::K_RADIX_PLAN_LIVE_N), K_LIVE_COUNT);
            EXPECT_EQ(word(Plan::K_RADIX_PLAN_INTO_INDICES), row.into_indices);
        }

        // The last active digit writes the indices, and the digits before it alternate backwards.
        INSTANTIATE_TEST_SUITE_P(
            , Small_GraphPrimitivesRadixPlanDigit,
            ::testing::Values(PlanRow{1, 0, true, 1u}, PlanRow{1, 1, false, 0u},
                              PlanRow{1, 2, false, 0u}, PlanRow{2, 0, true, 0u},
                              PlanRow{2, 1, true, 1u}, PlanRow{2, 2, false, 0u},
                              PlanRow{3, 0, true, 1u}, PlanRow{3, 1, true, 0u},
                              PlanRow{3, 2, true, 1u}));

    } // namespace Test
} // namespace SushiBLAS
