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
        } // namespace

        /** @brief Tests the digit count at the 8-bit boundaries of the key range. */
        TEST(Medium_GraphPrimitivesRadixPlan, DigitCountAtByteBoundaries)
        {
            EXPECT_EQ(radix_digit_count(0), 1u);
            EXPECT_EQ(radix_digit_count(1), 1u);
            EXPECT_EQ(radix_digit_count(std::size_t{1} << 8), 1u);
            EXPECT_EQ(radix_digit_count((std::size_t{1} << 8) + 1), 2u);
            EXPECT_EQ(radix_digit_count(std::size_t{1} << 16), 2u);
            EXPECT_EQ(radix_digit_count((std::size_t{1} << 16) + 1), 3u);
            EXPECT_EQ(radix_digit_count(std::size_t{1} << 24), 3u);
            EXPECT_EQ(radix_digit_count((std::size_t{1} << 24) + 1), 4u);
        }

        /** @brief Tests that each digit of a key is the byte at its shift. */
        TEST(Medium_GraphPrimitivesRadixPlan, DigitIsTheByteAtItsShift)
        {
            constexpr std::uint32_t K_KEY = 0xA1B2C3D4u;

            EXPECT_EQ(radix_digit(K_KEY, 0), 0xD4u);
            EXPECT_EQ(radix_digit(K_KEY, 8), 0xC3u);
            EXPECT_EQ(radix_digit(K_KEY, 16), 0xB2u);
            EXPECT_EQ(radix_digit(K_KEY, 24), 0xA1u);
        }

        /** @brief Tests that a three-digit plan runs only the digits the live key range needs. */
        TEST(Medium_GraphPrimitivesRadixPlan, PlanSkipsDigitsAboveTheLiveKeyRange)
        {
            constexpr std::size_t K_DIGITS = 3;
            constexpr std::size_t K_LIVE_COUNT = 1000;
            constexpr std::size_t K_LIVE_TILES = 4;
            std::vector<std::uint32_t> plan(K_DIGITS * Plan::K_RADIX_PLAN_WORDS, 0xFFFFFFFFu);

            const auto word = [&plan](std::size_t digit, std::size_t offset)
            {
                return plan[digit * Plan::K_RADIX_PLAN_WORDS + offset];
            };

            for (std::size_t live_digits = 1; live_digits <= K_DIGITS; ++live_digits)
            {
                const std::size_t live_key_range = std::size_t{1} << (8 * live_digits);
                fill_radix_plan(K_DIGITS, K_LIVE_COUNT, K_LIVE_TILES, live_key_range,
                                plan.data());
                for (std::size_t d = 0; d < K_DIGITS; ++d)
                {
                    const bool active = d < live_digits;
                    EXPECT_EQ(word(d, Plan::K_RADIX_PLAN_TILES), active ? K_LIVE_TILES : 0u)
                        << "live digits " << live_digits << ", d " << d;
                    EXPECT_EQ(word(d, Plan::K_RADIX_PLAN_BINS),
                              active ? Plan::K_RADIX_DIGIT_BINS : 0u);
                    EXPECT_EQ(word(d, Plan::K_RADIX_PLAN_ONE), active ? 1u : 0u);
                    EXPECT_EQ(word(d, Plan::K_RADIX_PLAN_LIVE_N), K_LIVE_COUNT);
                    const std::uint32_t expected_into =
                        (active && (live_digits - 1 - d) % 2 == 0) ? 1u : 0u;
                    EXPECT_EQ(word(d, Plan::K_RADIX_PLAN_INTO_INDICES), expected_into);
                }
                EXPECT_EQ(word(live_digits - 1, Plan::K_RADIX_PLAN_INTO_INDICES), 1u)
                    << "the last active digit must write the indices";
            }
        }

    } // namespace Test
} // namespace SushiBLAS
