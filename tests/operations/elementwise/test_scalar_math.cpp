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
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/arithmetic/safe_math.hpp>
#include <SushiBLAS/elementwise/dispatch.hpp>
#include <SushiBLAS/elementwise/scalar_math.hpp>
#include <SushiBLAS/functor_map/scalar.hpp>
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

        /// @brief A positive infinity of double.
        constexpr double K_INF = std::numeric_limits<double>::infinity();

        TEST(Small_ScalarMath, SineAndTangentOfOneReturnTheirSpecifiedValues)
        {
            expect_close(Dispatch::scalar_sin(1.0), 0.8414709848078965);
            expect_close(Dispatch::scalar_tan(1.0), 1.5574077246549023);
            EXPECT_EQ(Dispatch::scalar_sin(0.0), 0.0);
            EXPECT_EQ(Dispatch::scalar_tan(0.0), 0.0);
        }

        TEST(Small_ScalarMath, HyperbolicSineOfOneIsHalfTheDifferenceOfEAndItsInverse)
        {
            expect_close(Dispatch::scalar_sinh(1.0), 1.1752011936438014);
            EXPECT_EQ(Dispatch::scalar_sinh(0.0), 0.0);
        }

        TEST(Small_ScalarMath, SineAndCosineOfInfinityAreNaN)
        {
            EXPECT_TRUE(std::isnan(Dispatch::scalar_sin(K_INF)));
            EXPECT_TRUE(std::isnan(Dispatch::scalar_cos(K_INF)));
        }

        TEST(Small_ScalarMath, RoundSendsAHalfwayValueAwayFromZero)
        {
            EXPECT_EQ(Dispatch::scalar_round(2.5), 3.0);
            EXPECT_EQ(Dispatch::scalar_round(-2.5), -3.0);
            EXPECT_EQ(Dispatch::scalar_round(2.4), 2.0);
        }

        TEST(Small_ScalarMath, SquareMultipliesTheValueByItself)
        {
            EXPECT_EQ(Dispatch::scalar_square(-3.0), 9.0);
            EXPECT_EQ(Dispatch::scalar_square(0.5f), 0.25f);
        }

        TEST(Small_ScalarMath, ReciprocalDividesOneByTheValue)
        {
            EXPECT_EQ(Dispatch::scalar_reciprocal(4.0), 0.25);
            EXPECT_TRUE(std::isinf(Dispatch::scalar_reciprocal(0.0)));
        }

        TEST(Small_ScalarMath, PowRaisesTheBaseToTheExponent)
        {
            EXPECT_EQ(Dispatch::scalar_pow(2.0, 10.0), 1024.0);
            EXPECT_EQ(Dispatch::scalar_pow(9.0, 0.5), 3.0);
        }

        TEST(Small_ScalarMath, RemainderRoundsTheQuotientToTheNearestEven)
        {
            // 5.5 / 2 = 2.75 rounds to 3, so the remainder is 5.5 - 6 = -0.5.
            EXPECT_EQ(Dispatch::scalar_remainder(5.5, 2.0), -0.5);
            // 5 / 2 = 2.5 rounds to the even 2, so the remainder is 1.
            EXPECT_EQ(Dispatch::scalar_remainder(5.0, 2.0), 1.0);
        }

        TEST(Small_ScalarMath, ExponentialOfZeroIsOneAndOfInfinityIsInfinity)
        {
            EXPECT_EQ(Dispatch::safe_exp(0.0), 1.0);
            EXPECT_EQ(Dispatch::safe_exp(K_INF), K_INF);
        }

        TEST(Small_ScalarMath, LogOfZeroIsMinusInfinityAndOfInfinityIsInfinity)
        {
            EXPECT_EQ(Dispatch::safe_log(0.0), -K_INF);
            EXPECT_EQ(Dispatch::safe_log(K_INF), K_INF);
        }

        TEST(Small_ScalarMath, LogOfANegativeNumberIsNaN)
        {
            EXPECT_TRUE(std::isnan(Dispatch::safe_log(-1.0)));
        }

        TEST(Small_ScalarMath, SquareRootOfANegativeNumberIsNaN)
        {
            EXPECT_TRUE(std::isnan(Dispatch::safe_sqrt(-1.0)));
        }

        TEST(Small_ScalarMath, SquareRootOfZeroIsZero)
        {
            EXPECT_EQ(Dispatch::safe_sqrt(0.0), 0.0);
        }

        TEST(Small_ScalarMath, SquareRootOfInfinityIsInfinity)
        {
            EXPECT_EQ(Dispatch::safe_sqrt(K_INF), K_INF);
        }

        TEST(Small_ScalarMath, ClampWithEqualBoundsGivesThatBound)
        {
            // Equal bounds pass the record-time check, or the call throws and the test fails.
            Dispatch::require_clamp_bounds(2.0, 2.0);
            EXPECT_EQ(Dispatch::scalar_clamp(-5.0, 2.0, 2.0), 2.0);
            EXPECT_EQ(Dispatch::scalar_clamp(9.0, 2.0, 2.0), 2.0);
        }

        TEST(Small_ScalarMath, ClampBoundsTruncateTowardZeroOnAnIntegerElement)
        {
            // A lower bound of -2.5 truncates to -2 and an upper bound of 7.9 truncates to 7.
            const Dispatch::ElementScalar low(-2.5);
            const Dispatch::ElementScalar high(7.9);

            EXPECT_EQ(Dispatch::scalar_clamp(int32_t{-10}, low.as<int32_t>(), high.as<int32_t>()), -2);
            EXPECT_EQ(Dispatch::scalar_clamp(int32_t{0}, low.as<int32_t>(), high.as<int32_t>()), 0);
            EXPECT_EQ(Dispatch::scalar_clamp(int32_t{10}, low.as<int32_t>(), high.as<int32_t>()), 7);
        }

        TEST(Small_ScalarMath, ClampKeepsADoubleBoundThatFloatWouldDrop)
        {
            // 1 + 1e-9 rounds to exactly 1.0f in float and is kept in double.
            constexpr double K_FINE_LOW = 1.0 + 1e-9;
            const Dispatch::ElementScalar low(K_FINE_LOW);

            EXPECT_EQ(Dispatch::scalar_clamp(1.0, low.as<double>(), 2.0), K_FINE_LOW);
        }

        TEST(Small_ScalarMath, ClampBoundsRefuseALowerBoundAboveTheUpperOne)
        {
            EXPECT_THROW(Dispatch::require_clamp_bounds(7.9, -2.5), std::invalid_argument);
        }

    } // namespace Test
} // namespace SushiBLAS
