/**************************************************************************/
/* test_scalar_math.cpp                                                   */
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
 * @file test_scalar_math.cpp
 * @brief Tests the host calls of the scalar functions elementwise applies to one element.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <gtest/gtest.h>
#include <SushiBLAS/elementwise/scalar_math.hpp>
#include "support/tolerance.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief A quiet NaN of double.
        constexpr double K_NAN = std::numeric_limits<double>::quiet_NaN();

        /// @brief Expects a double result within the derived bound of one rounded operation.
        static void expect_close(double computed, double reference)
        {
            EXPECT_TRUE(TestSupport::is_within(
                TestSupport::measure_relative_error<double>(computed, reference, 1)));
        }

        TEST(Small_ScalarMath, InverseTrigonometricFunctionsReturnTheirSpecifiedValues)
        {
            // acos(0.5) = pi / 3, asin(0.5) = pi / 6, atan(1) = pi / 4.
            expect_close(Dispatch::scalar_acos(0.5), 1.0471975511965976);
            expect_close(Dispatch::scalar_asin(0.5), 0.5235987755982988);
            expect_close(Dispatch::scalar_atan(1.0), 0.7853981633974483);
            EXPECT_EQ(Dispatch::scalar_acos(1.0), 0.0);
            EXPECT_EQ(Dispatch::scalar_asin(0.0), 0.0);
            EXPECT_EQ(Dispatch::scalar_atan(0.0), 0.0);
        }

        TEST(Small_ScalarMath, HyperbolicFunctionsReturnTheirSpecifiedValues)
        {
            // cosh(1) = (e + 1/e) / 2.
            expect_close(Dispatch::scalar_cosh(1.0), 1.5430806348152437);
            // acosh(2) = ln(2 + sqrt(3)).
            expect_close(Dispatch::scalar_acosh(2.0), 1.3169578969248166);
            // asinh(1) = ln(1 + sqrt(2)).
            expect_close(Dispatch::scalar_asinh(1.0), 0.881373587019543);
            // atanh(0.5) = ln(3) / 2.
            expect_close(Dispatch::scalar_atanh(0.5), 0.5493061443340548);
            EXPECT_EQ(Dispatch::scalar_cosh(0.0), 1.0);
            EXPECT_EQ(Dispatch::scalar_acosh(1.0), 0.0);
            EXPECT_EQ(Dispatch::scalar_asinh(0.0), 0.0);
            EXPECT_EQ(Dispatch::scalar_atanh(0.0), 0.0);
        }

        TEST(Small_ScalarMath, CosineOfZeroIsOne)
        {
            EXPECT_EQ(Dispatch::scalar_cos(0.0), 1.0);
            EXPECT_EQ(Dispatch::scalar_cos(0.0f), 1.0f);
        }

        TEST(Small_ScalarMath, CeilAndFloorRoundTowardsTheirInfinity)
        {
            EXPECT_EQ(Dispatch::scalar_ceil(1.25), 2.0);
            EXPECT_EQ(Dispatch::scalar_ceil(-1.25), -1.0);
            EXPECT_EQ(Dispatch::scalar_floor(1.25), 1.0);
            EXPECT_EQ(Dispatch::scalar_floor(-1.25), -2.0);
        }

        TEST(Small_ScalarMath, DivisionFollowsTheBuiltInOperator)
        {
            EXPECT_EQ(Dispatch::scalar_div(7.0, 2.0), 3.5);
            EXPECT_EQ(Dispatch::scalar_div(7, 2), 3);
            EXPECT_EQ(Dispatch::scalar_div(-7, 2), -3);
            EXPECT_TRUE(std::isinf(Dispatch::scalar_div(1.0, 0.0)));
        }

        TEST(Small_ScalarMath, FmodKeepsTheSignOfTheDividend)
        {
            EXPECT_EQ(Dispatch::scalar_fmod(5.5, 2.0), 1.5);
            EXPECT_EQ(Dispatch::scalar_fmod(-5.5, 2.0), -1.5);
        }

        TEST(Small_ScalarMath, ClampLimitsAValueToTheClosedInterval)
        {
            EXPECT_EQ(Dispatch::scalar_clamp(5.0, 0.0, 3.0), 3.0);
            EXPECT_EQ(Dispatch::scalar_clamp(-1.0, 0.0, 3.0), 0.0);
            EXPECT_EQ(Dispatch::scalar_clamp(2.0, 0.0, 3.0), 2.0);
        }

        TEST(Small_ScalarMath, MaxAndMinOfFloatingValuesPickTheLargerAndTheSmaller)
        {
            EXPECT_EQ(Dispatch::scalar_max(1.5, 2.5), 2.5);
            EXPECT_EQ(Dispatch::scalar_max(2.5f, -1.0f), 2.5f);
            EXPECT_EQ(Dispatch::scalar_min(1.5, 2.5), 1.5);
            EXPECT_EQ(Dispatch::scalar_min(2.5f, -1.0f), -1.0f);
        }

        TEST(Small_ScalarMath, MaxAndMinReturnNaNWhenTheFirstOperandIsNaN)
        {
            EXPECT_TRUE(std::isnan(Dispatch::scalar_max(K_NAN, 1.0)));
            EXPECT_TRUE(std::isnan(Dispatch::scalar_min(K_NAN, 1.0)));
        }

        TEST(Small_ScalarMath, MaxAndMinReturnNaNWhenTheSecondOperandIsNaN)
        {
            EXPECT_TRUE(std::isnan(Dispatch::scalar_max(1.0, K_NAN)));
            EXPECT_TRUE(std::isnan(Dispatch::scalar_min(1.0, K_NAN)));
        }

        TEST(Small_ScalarMath, MaxAndMinOfIntegersTakeTheIntegralBranch)
        {
            EXPECT_EQ(Dispatch::scalar_max(3, 7), 7);
            EXPECT_EQ(Dispatch::scalar_max(int64_t{-3}, int64_t{-7}), int64_t{-3});
            EXPECT_EQ(Dispatch::scalar_min(3, 7), 3);
            EXPECT_EQ(Dispatch::scalar_min(int64_t{-3}, int64_t{-7}), int64_t{-7});
        }
    } // namespace Test
} // namespace SushiBLAS
