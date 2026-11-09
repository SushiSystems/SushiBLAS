/**************************************************************************/
/* test_wrapping_math.cpp                                                 */
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
 * @file test_wrapping_math.cpp
 * @brief Tests that integer arithmetic wraps modulo its width and floating arithmetic is plain.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <limits>
#include <gtest/gtest.h>
#include <SushiBLAS/elementwise/wrapping_math.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The largest INT32.
        constexpr int32_t K_MAX32 = std::numeric_limits<int32_t>::max();

        /// @brief The smallest INT32.
        constexpr int32_t K_MIN32 = std::numeric_limits<int32_t>::min();

        TEST(Small_WrappingMath, AddingOneToTheLargestInt32GivesTheSmallest)
        {
            EXPECT_EQ(Dispatch::wrapping_add(K_MAX32, int32_t{1}), K_MIN32);
        }

        TEST(Small_WrappingMath, SubtractingOneFromTheSmallestInt32GivesTheLargest)
        {
            EXPECT_EQ(Dispatch::wrapping_sub(K_MIN32, int32_t{1}), K_MAX32);
        }

        TEST(Small_WrappingMath, DoublingTheLargestInt32GivesMinusTwo)
        {
            EXPECT_EQ(Dispatch::wrapping_mul(K_MAX32, int32_t{2}), int32_t{-2});
        }

        TEST(Small_WrappingMath, TheSmallestInt32TimesMinusOneIsItself)
        {
            EXPECT_EQ(Dispatch::wrapping_mul(K_MIN32, int32_t{-1}), K_MIN32);
        }

        TEST(Small_WrappingMath, NegatingTheSmallestInt32GivesItself)
        {
            EXPECT_EQ(Dispatch::wrapping_neg(K_MIN32), K_MIN32);
        }

        TEST(Small_WrappingMath, TheAbsoluteValueOfTheSmallestInt32IsItself)
        {
            EXPECT_EQ(Dispatch::wrapping_abs(K_MIN32), K_MIN32);
        }

        TEST(Small_WrappingMath, TheAbsoluteValueOfANegativeInt64IsPositive)
        {
            EXPECT_EQ(Dispatch::wrapping_abs(int64_t{-5}), int64_t{5});
        }

        TEST(Small_WrappingMath, FloatingAdditionIsPlain)
        {
            EXPECT_EQ(Dispatch::wrapping_add(1.5, 2.25), 3.75);
        }

        TEST(Small_WrappingMath, FloatingSubtractionIsPlain)
        {
            EXPECT_EQ(Dispatch::wrapping_sub(1.0f, 4.0f), -3.0f);
        }

        TEST(Small_WrappingMath, FloatingMultiplicationIsPlain)
        {
            EXPECT_EQ(Dispatch::wrapping_mul(-2.0, 3.0), -6.0);
        }

        TEST(Small_WrappingMath, FloatingNegationFlipsTheSign)
        {
            EXPECT_EQ(Dispatch::wrapping_neg(2.5), -2.5);
        }

        TEST(Small_WrappingMath, TheFloatingAbsoluteValueDropsTheSign)
        {
            EXPECT_EQ(Dispatch::wrapping_abs(-2.5f), 2.5f);
        }
    } // namespace Test
} // namespace SushiBLAS
