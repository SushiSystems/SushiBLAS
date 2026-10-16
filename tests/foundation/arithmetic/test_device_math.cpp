/**************************************************************************/
/* test_device_math.cpp                                                   */
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
 * @file test_device_math.cpp
 * @brief Tests that the host side of Math names the standard functions with their results.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <gtest/gtest.h>
#include <SushiBLAS/arithmetic/device_math.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Absolute tolerance of a double result against its hand value.
        constexpr double K_TOLERANCE = 1e-12;

        TEST(Medium_DeviceMath, TheHostFunctionsGiveTheStandardResults)
        {
            EXPECT_EQ(Math::abs(-2.5), 2.5);
            EXPECT_EQ(Math::floor(-1.5), -2.0);
            EXPECT_EQ(Math::sqrt(16.0), 4.0);
            EXPECT_NEAR(Math::cbrt(27.0), 3.0, K_TOLERANCE);
            EXPECT_EQ(Math::copysign(3.0, -0.0), -3.0);
            EXPECT_EQ(Math::fmod(7.0, 3.0), 1.0);
            EXPECT_NEAR(Math::pow(2.0, 10.0), 1024.0, K_TOLERANCE);
            EXPECT_NEAR(Math::exp(1.0), 2.718281828459045, K_TOLERANCE);
            EXPECT_NEAR(Math::log(2.718281828459045), 1.0, K_TOLERANCE);
            EXPECT_NEAR(Math::sin(0.5), std::sin(0.5), K_TOLERANCE);
            EXPECT_NEAR(Math::cos(0.5), std::cos(0.5), K_TOLERANCE);
            EXPECT_NEAR(Math::asin(1.0), 1.5707963267948966, K_TOLERANCE);
            EXPECT_NEAR(Math::atan2(1.0, -1.0), 2.356194490192345, K_TOLERANCE);
        }

        TEST(Medium_DeviceMath, IsFiniteSeparatesFiniteFromInfinityAndNaN)
        {
            EXPECT_TRUE(Math::isfinite(1.0f));
            EXPECT_FALSE(Math::isfinite(std::numeric_limits<float>::infinity()));
            EXPECT_FALSE(Math::isfinite(std::numeric_limits<double>::quiet_NaN()));
        }

        TEST(Medium_DeviceMath, AbsKeepsTheTypeOfItsArgument)
        {
            EXPECT_EQ(Math::abs(-3.0f), 3.0f);
            EXPECT_EQ(Math::abs(-0.0), 0.0);
        }
    } // namespace Test
} // namespace SushiBLAS
