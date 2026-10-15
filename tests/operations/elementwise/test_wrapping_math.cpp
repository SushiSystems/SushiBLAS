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

        TEST(Small_WrappingMath, IntegersWrapAroundTheirWidth)
        {
            EXPECT_EQ(Dispatch::wrapping_add(K_MAX32, int32_t{1}), K_MIN32);
            EXPECT_EQ(Dispatch::wrapping_sub(K_MIN32, int32_t{1}), K_MAX32);
            EXPECT_EQ(Dispatch::wrapping_mul(K_MAX32, int32_t{2}), int32_t{-2});
            EXPECT_EQ(Dispatch::wrapping_neg(K_MIN32), K_MIN32);
            EXPECT_EQ(Dispatch::wrapping_abs(K_MIN32), K_MIN32);
            EXPECT_EQ(Dispatch::wrapping_abs(int64_t{-5}), int64_t{5});
        }

        TEST(Small_WrappingMath, FloatingValuesFollowPlainArithmetic)
        {
            EXPECT_EQ(Dispatch::wrapping_add(1.5, 2.25), 3.75);
            EXPECT_EQ(Dispatch::wrapping_sub(1.0f, 4.0f), -3.0f);
            EXPECT_EQ(Dispatch::wrapping_mul(-2.0, 3.0), -6.0);
            EXPECT_EQ(Dispatch::wrapping_neg(2.5), -2.5);
            EXPECT_EQ(Dispatch::wrapping_abs(-2.5f), 2.5f);
        }
    } // namespace Test
} // namespace SushiBLAS
