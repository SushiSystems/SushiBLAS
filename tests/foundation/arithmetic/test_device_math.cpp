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
 * @file test_device_math.cpp
 * @brief Tests that the host side of Math names the standard functions with their results.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/arithmetic/device_math.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include "support/tolerance.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Absolute tolerance: 1024 double roundings of a value of order one.
        constexpr double K_TOLERANCE =
            TestSupport::K_DEFAULT_THRESHOLD * 1024.0 * TestSupport::unit_roundoff<double>();

        /// @brief Reports whether two results carry the same bits, or are both NaN.
        template<typename T>
        bool same_bits(T left, T right)
        {
            if (std::isnan(left) || std::isnan(right))
            {
                return std::isnan(left) && std::isnan(right);
            }
            return std::memcmp(&left, &right, sizeof(T)) == 0;
        }

        /// @brief Floating-point types whose host Math functions are checked against std.
        using MathTypes = ::testing::Types<float, double>;

        /// @brief Typed fixture for the host Math functions on one floating-point type.
        template<typename T>
        class Small_DeviceMath : public ::testing::Test
        {
        };
        SB_TYPED_TEST_SUITE(Small_DeviceMath, MathTypes);

        TYPED_TEST(Small_DeviceMath, SqrtMatchesStdBitForBit)
        {
            using T = TypeParam;
            EXPECT_TRUE(same_bits<T>(Math::sqrt(T(0)), std::sqrt(T(0))));
            EXPECT_TRUE(same_bits<T>(Math::sqrt(-T(0)), std::sqrt(-T(0))));
            EXPECT_TRUE(same_bits<T>(Math::sqrt(std::numeric_limits<T>::min()), std::sqrt(std::numeric_limits<T>::min())));
            EXPECT_TRUE(same_bits<T>(Math::sqrt(std::numeric_limits<T>::denorm_min()), std::sqrt(std::numeric_limits<T>::denorm_min())));
            EXPECT_TRUE(same_bits<T>(Math::sqrt(-std::numeric_limits<T>::denorm_min()), std::sqrt(-std::numeric_limits<T>::denorm_min())));
            EXPECT_TRUE(same_bits<T>(Math::sqrt(std::numeric_limits<T>::max()), std::sqrt(std::numeric_limits<T>::max())));
            EXPECT_TRUE(same_bits<T>(Math::sqrt(-std::numeric_limits<T>::max()), std::sqrt(-std::numeric_limits<T>::max())));
            EXPECT_TRUE(same_bits<T>(Math::sqrt(std::numeric_limits<T>::infinity()), std::sqrt(std::numeric_limits<T>::infinity())));
            EXPECT_TRUE(same_bits<T>(Math::sqrt(-std::numeric_limits<T>::infinity()), std::sqrt(-std::numeric_limits<T>::infinity())));
            EXPECT_TRUE(same_bits<T>(Math::sqrt(std::numeric_limits<T>::quiet_NaN()), std::sqrt(std::numeric_limits<T>::quiet_NaN())));
            EXPECT_TRUE(same_bits<T>(Math::sqrt(T(1)), std::sqrt(T(1))));
            EXPECT_TRUE(same_bits<T>(Math::sqrt(T(-1)), std::sqrt(T(-1))));
            EXPECT_TRUE(same_bits<T>(Math::sqrt(T(0.5)), std::sqrt(T(0.5))));
            EXPECT_TRUE(same_bits<T>(Math::sqrt(T(-2.5)), std::sqrt(T(-2.5))));
        }

        TYPED_TEST(Small_DeviceMath, FloorMatchesStdBitForBit)
        {
            using T = TypeParam;
            EXPECT_TRUE(same_bits<T>(Math::floor(T(0)), std::floor(T(0))));
            EXPECT_TRUE(same_bits<T>(Math::floor(-T(0)), std::floor(-T(0))));
            EXPECT_TRUE(same_bits<T>(Math::floor(std::numeric_limits<T>::min()), std::floor(std::numeric_limits<T>::min())));
            EXPECT_TRUE(same_bits<T>(Math::floor(std::numeric_limits<T>::denorm_min()), std::floor(std::numeric_limits<T>::denorm_min())));
            EXPECT_TRUE(same_bits<T>(Math::floor(-std::numeric_limits<T>::denorm_min()), std::floor(-std::numeric_limits<T>::denorm_min())));
            EXPECT_TRUE(same_bits<T>(Math::floor(std::numeric_limits<T>::max()), std::floor(std::numeric_limits<T>::max())));
            EXPECT_TRUE(same_bits<T>(Math::floor(-std::numeric_limits<T>::max()), std::floor(-std::numeric_limits<T>::max())));
            EXPECT_TRUE(same_bits<T>(Math::floor(std::numeric_limits<T>::infinity()), std::floor(std::numeric_limits<T>::infinity())));
            EXPECT_TRUE(same_bits<T>(Math::floor(-std::numeric_limits<T>::infinity()), std::floor(-std::numeric_limits<T>::infinity())));
            EXPECT_TRUE(same_bits<T>(Math::floor(std::numeric_limits<T>::quiet_NaN()), std::floor(std::numeric_limits<T>::quiet_NaN())));
            EXPECT_TRUE(same_bits<T>(Math::floor(T(1)), std::floor(T(1))));
            EXPECT_TRUE(same_bits<T>(Math::floor(T(-1)), std::floor(T(-1))));
            EXPECT_TRUE(same_bits<T>(Math::floor(T(0.5)), std::floor(T(0.5))));
            EXPECT_TRUE(same_bits<T>(Math::floor(T(-2.5)), std::floor(T(-2.5))));
        }

        TYPED_TEST(Small_DeviceMath, AbsMatchesStdBitForBit)
        {
            using T = TypeParam;
            EXPECT_TRUE(same_bits<T>(Math::abs(T(0)), std::abs(T(0))));
            EXPECT_TRUE(same_bits<T>(Math::abs(-T(0)), std::abs(-T(0))));
            EXPECT_TRUE(same_bits<T>(Math::abs(std::numeric_limits<T>::min()), std::abs(std::numeric_limits<T>::min())));
            EXPECT_TRUE(same_bits<T>(Math::abs(std::numeric_limits<T>::denorm_min()), std::abs(std::numeric_limits<T>::denorm_min())));
            EXPECT_TRUE(same_bits<T>(Math::abs(-std::numeric_limits<T>::denorm_min()), std::abs(-std::numeric_limits<T>::denorm_min())));
            EXPECT_TRUE(same_bits<T>(Math::abs(std::numeric_limits<T>::max()), std::abs(std::numeric_limits<T>::max())));
            EXPECT_TRUE(same_bits<T>(Math::abs(-std::numeric_limits<T>::max()), std::abs(-std::numeric_limits<T>::max())));
            EXPECT_TRUE(same_bits<T>(Math::abs(std::numeric_limits<T>::infinity()), std::abs(std::numeric_limits<T>::infinity())));
            EXPECT_TRUE(same_bits<T>(Math::abs(-std::numeric_limits<T>::infinity()), std::abs(-std::numeric_limits<T>::infinity())));
            EXPECT_TRUE(same_bits<T>(Math::abs(std::numeric_limits<T>::quiet_NaN()), std::abs(std::numeric_limits<T>::quiet_NaN())));
            EXPECT_TRUE(same_bits<T>(Math::abs(T(1)), std::abs(T(1))));
            EXPECT_TRUE(same_bits<T>(Math::abs(T(-1)), std::abs(T(-1))));
            EXPECT_TRUE(same_bits<T>(Math::abs(T(0.5)), std::abs(T(0.5))));
            EXPECT_TRUE(same_bits<T>(Math::abs(T(-2.5)), std::abs(T(-2.5))));
        }

        TYPED_TEST(Small_DeviceMath, AsinMatchesStdBitForBit)
        {
            using T = TypeParam;
            EXPECT_TRUE(same_bits<T>(Math::asin(T(0)), std::asin(T(0))));
            EXPECT_TRUE(same_bits<T>(Math::asin(-T(0)), std::asin(-T(0))));
            EXPECT_TRUE(same_bits<T>(Math::asin(std::numeric_limits<T>::min()), std::asin(std::numeric_limits<T>::min())));
            EXPECT_TRUE(same_bits<T>(Math::asin(std::numeric_limits<T>::denorm_min()), std::asin(std::numeric_limits<T>::denorm_min())));
            EXPECT_TRUE(same_bits<T>(Math::asin(-std::numeric_limits<T>::denorm_min()), std::asin(-std::numeric_limits<T>::denorm_min())));
            EXPECT_TRUE(same_bits<T>(Math::asin(std::numeric_limits<T>::max()), std::asin(std::numeric_limits<T>::max())));
            EXPECT_TRUE(same_bits<T>(Math::asin(-std::numeric_limits<T>::max()), std::asin(-std::numeric_limits<T>::max())));
            EXPECT_TRUE(same_bits<T>(Math::asin(std::numeric_limits<T>::infinity()), std::asin(std::numeric_limits<T>::infinity())));
            EXPECT_TRUE(same_bits<T>(Math::asin(-std::numeric_limits<T>::infinity()), std::asin(-std::numeric_limits<T>::infinity())));
            EXPECT_TRUE(same_bits<T>(Math::asin(std::numeric_limits<T>::quiet_NaN()), std::asin(std::numeric_limits<T>::quiet_NaN())));
            EXPECT_TRUE(same_bits<T>(Math::asin(T(1)), std::asin(T(1))));
            EXPECT_TRUE(same_bits<T>(Math::asin(T(-1)), std::asin(T(-1))));
            EXPECT_TRUE(same_bits<T>(Math::asin(T(0.5)), std::asin(T(0.5))));
            EXPECT_TRUE(same_bits<T>(Math::asin(T(-2.5)), std::asin(T(-2.5))));
        }

        TYPED_TEST(Small_DeviceMath, CbrtMatchesStdBitForBit)
        {
            using T = TypeParam;
            EXPECT_TRUE(same_bits<T>(Math::cbrt(T(0)), std::cbrt(T(0))));
            EXPECT_TRUE(same_bits<T>(Math::cbrt(-T(0)), std::cbrt(-T(0))));
            EXPECT_TRUE(same_bits<T>(Math::cbrt(std::numeric_limits<T>::min()), std::cbrt(std::numeric_limits<T>::min())));
            EXPECT_TRUE(same_bits<T>(Math::cbrt(std::numeric_limits<T>::denorm_min()), std::cbrt(std::numeric_limits<T>::denorm_min())));
            EXPECT_TRUE(same_bits<T>(Math::cbrt(-std::numeric_limits<T>::denorm_min()), std::cbrt(-std::numeric_limits<T>::denorm_min())));
            EXPECT_TRUE(same_bits<T>(Math::cbrt(std::numeric_limits<T>::max()), std::cbrt(std::numeric_limits<T>::max())));
            EXPECT_TRUE(same_bits<T>(Math::cbrt(-std::numeric_limits<T>::max()), std::cbrt(-std::numeric_limits<T>::max())));
            EXPECT_TRUE(same_bits<T>(Math::cbrt(std::numeric_limits<T>::infinity()), std::cbrt(std::numeric_limits<T>::infinity())));
            EXPECT_TRUE(same_bits<T>(Math::cbrt(-std::numeric_limits<T>::infinity()), std::cbrt(-std::numeric_limits<T>::infinity())));
            EXPECT_TRUE(same_bits<T>(Math::cbrt(std::numeric_limits<T>::quiet_NaN()), std::cbrt(std::numeric_limits<T>::quiet_NaN())));
            EXPECT_TRUE(same_bits<T>(Math::cbrt(T(1)), std::cbrt(T(1))));
            EXPECT_TRUE(same_bits<T>(Math::cbrt(T(-1)), std::cbrt(T(-1))));
            EXPECT_TRUE(same_bits<T>(Math::cbrt(T(0.5)), std::cbrt(T(0.5))));
            EXPECT_TRUE(same_bits<T>(Math::cbrt(T(-2.5)), std::cbrt(T(-2.5))));
        }

        TYPED_TEST(Small_DeviceMath, ExpMatchesStdBitForBit)
        {
            using T = TypeParam;
            EXPECT_TRUE(same_bits<T>(Math::exp(T(0)), std::exp(T(0))));
            EXPECT_TRUE(same_bits<T>(Math::exp(-T(0)), std::exp(-T(0))));
            EXPECT_TRUE(same_bits<T>(Math::exp(std::numeric_limits<T>::min()), std::exp(std::numeric_limits<T>::min())));
            EXPECT_TRUE(same_bits<T>(Math::exp(std::numeric_limits<T>::denorm_min()), std::exp(std::numeric_limits<T>::denorm_min())));
            EXPECT_TRUE(same_bits<T>(Math::exp(-std::numeric_limits<T>::denorm_min()), std::exp(-std::numeric_limits<T>::denorm_min())));
            EXPECT_TRUE(same_bits<T>(Math::exp(std::numeric_limits<T>::max()), std::exp(std::numeric_limits<T>::max())));
            EXPECT_TRUE(same_bits<T>(Math::exp(-std::numeric_limits<T>::max()), std::exp(-std::numeric_limits<T>::max())));
            EXPECT_TRUE(same_bits<T>(Math::exp(std::numeric_limits<T>::infinity()), std::exp(std::numeric_limits<T>::infinity())));
            EXPECT_TRUE(same_bits<T>(Math::exp(-std::numeric_limits<T>::infinity()), std::exp(-std::numeric_limits<T>::infinity())));
            EXPECT_TRUE(same_bits<T>(Math::exp(std::numeric_limits<T>::quiet_NaN()), std::exp(std::numeric_limits<T>::quiet_NaN())));
            EXPECT_TRUE(same_bits<T>(Math::exp(T(1)), std::exp(T(1))));
            EXPECT_TRUE(same_bits<T>(Math::exp(T(-1)), std::exp(T(-1))));
            EXPECT_TRUE(same_bits<T>(Math::exp(T(0.5)), std::exp(T(0.5))));
            EXPECT_TRUE(same_bits<T>(Math::exp(T(-2.5)), std::exp(T(-2.5))));
        }

        TYPED_TEST(Small_DeviceMath, LogMatchesStdBitForBit)
        {
            using T = TypeParam;
            EXPECT_TRUE(same_bits<T>(Math::log(T(0)), std::log(T(0))));
            EXPECT_TRUE(same_bits<T>(Math::log(-T(0)), std::log(-T(0))));
            EXPECT_TRUE(same_bits<T>(Math::log(std::numeric_limits<T>::min()), std::log(std::numeric_limits<T>::min())));
            EXPECT_TRUE(same_bits<T>(Math::log(std::numeric_limits<T>::denorm_min()), std::log(std::numeric_limits<T>::denorm_min())));
            EXPECT_TRUE(same_bits<T>(Math::log(-std::numeric_limits<T>::denorm_min()), std::log(-std::numeric_limits<T>::denorm_min())));
            EXPECT_TRUE(same_bits<T>(Math::log(std::numeric_limits<T>::max()), std::log(std::numeric_limits<T>::max())));
            EXPECT_TRUE(same_bits<T>(Math::log(-std::numeric_limits<T>::max()), std::log(-std::numeric_limits<T>::max())));
            EXPECT_TRUE(same_bits<T>(Math::log(std::numeric_limits<T>::infinity()), std::log(std::numeric_limits<T>::infinity())));
            EXPECT_TRUE(same_bits<T>(Math::log(-std::numeric_limits<T>::infinity()), std::log(-std::numeric_limits<T>::infinity())));
            EXPECT_TRUE(same_bits<T>(Math::log(std::numeric_limits<T>::quiet_NaN()), std::log(std::numeric_limits<T>::quiet_NaN())));
            EXPECT_TRUE(same_bits<T>(Math::log(T(1)), std::log(T(1))));
            EXPECT_TRUE(same_bits<T>(Math::log(T(-1)), std::log(T(-1))));
            EXPECT_TRUE(same_bits<T>(Math::log(T(0.5)), std::log(T(0.5))));
            EXPECT_TRUE(same_bits<T>(Math::log(T(-2.5)), std::log(T(-2.5))));
        }

        TYPED_TEST(Small_DeviceMath, IsFiniteMatchesStdBitForBit)
        {
            using T = TypeParam;
            EXPECT_EQ(Math::isfinite(T(0)), std::isfinite(T(0)));
            EXPECT_EQ(Math::isfinite(-T(0)), std::isfinite(-T(0)));
            EXPECT_EQ(Math::isfinite(std::numeric_limits<T>::min()), std::isfinite(std::numeric_limits<T>::min()));
            EXPECT_EQ(Math::isfinite(std::numeric_limits<T>::denorm_min()), std::isfinite(std::numeric_limits<T>::denorm_min()));
            EXPECT_EQ(Math::isfinite(-std::numeric_limits<T>::denorm_min()), std::isfinite(-std::numeric_limits<T>::denorm_min()));
            EXPECT_EQ(Math::isfinite(std::numeric_limits<T>::max()), std::isfinite(std::numeric_limits<T>::max()));
            EXPECT_EQ(Math::isfinite(-std::numeric_limits<T>::max()), std::isfinite(-std::numeric_limits<T>::max()));
            EXPECT_EQ(Math::isfinite(std::numeric_limits<T>::infinity()), std::isfinite(std::numeric_limits<T>::infinity()));
            EXPECT_EQ(Math::isfinite(-std::numeric_limits<T>::infinity()), std::isfinite(-std::numeric_limits<T>::infinity()));
            EXPECT_EQ(Math::isfinite(std::numeric_limits<T>::quiet_NaN()), std::isfinite(std::numeric_limits<T>::quiet_NaN()));
            EXPECT_EQ(Math::isfinite(T(1)), std::isfinite(T(1)));
            EXPECT_EQ(Math::isfinite(T(-1)), std::isfinite(T(-1)));
            EXPECT_EQ(Math::isfinite(T(0.5)), std::isfinite(T(0.5)));
            EXPECT_EQ(Math::isfinite(T(-2.5)), std::isfinite(T(-2.5)));
        }

        TYPED_TEST(Small_DeviceMath, FmodMatchesStdBitForBit)
        {
            using T = TypeParam;
            EXPECT_TRUE(same_bits<T>(Math::fmod(T(7), T(3)), std::fmod(T(7), T(3))));
            EXPECT_TRUE(same_bits<T>(Math::fmod(T(-7), T(3)), std::fmod(T(-7), T(3))));
            EXPECT_TRUE(same_bits<T>(Math::fmod(T(0.5), T(-0.0)), std::fmod(T(0.5), T(-0.0))));
            EXPECT_TRUE(same_bits<T>(Math::fmod(T(-2.5), T(0)), std::fmod(T(-2.5), T(0))));
            EXPECT_TRUE(same_bits<T>(Math::fmod(std::numeric_limits<T>::max(), T(0.5)), std::fmod(std::numeric_limits<T>::max(), T(0.5))));
            EXPECT_TRUE(same_bits<T>(Math::fmod(std::numeric_limits<T>::infinity(), T(2)), std::fmod(std::numeric_limits<T>::infinity(), T(2))));
            EXPECT_TRUE(same_bits<T>(Math::fmod(std::numeric_limits<T>::quiet_NaN(), T(1)), std::fmod(std::numeric_limits<T>::quiet_NaN(), T(1))));
            EXPECT_TRUE(same_bits<T>(Math::fmod(T(1), std::numeric_limits<T>::denorm_min()), std::fmod(T(1), std::numeric_limits<T>::denorm_min())));
            EXPECT_TRUE(same_bits<T>(Math::fmod(-T(0), T(-3)), std::fmod(-T(0), T(-3))));
            EXPECT_TRUE(same_bits<T>(Math::fmod(T(1), -T(1)), std::fmod(T(1), -T(1))));
        }

        TYPED_TEST(Small_DeviceMath, CopysignMatchesStdBitForBit)
        {
            using T = TypeParam;
            EXPECT_TRUE(same_bits<T>(Math::copysign(T(7), T(3)), std::copysign(T(7), T(3))));
            EXPECT_TRUE(same_bits<T>(Math::copysign(T(-7), T(3)), std::copysign(T(-7), T(3))));
            EXPECT_TRUE(same_bits<T>(Math::copysign(T(0.5), T(-0.0)), std::copysign(T(0.5), T(-0.0))));
            EXPECT_TRUE(same_bits<T>(Math::copysign(T(-2.5), T(0)), std::copysign(T(-2.5), T(0))));
            EXPECT_TRUE(same_bits<T>(Math::copysign(std::numeric_limits<T>::max(), T(0.5)), std::copysign(std::numeric_limits<T>::max(), T(0.5))));
            EXPECT_TRUE(same_bits<T>(Math::copysign(std::numeric_limits<T>::infinity(), T(2)), std::copysign(std::numeric_limits<T>::infinity(), T(2))));
            EXPECT_TRUE(same_bits<T>(Math::copysign(std::numeric_limits<T>::quiet_NaN(), T(1)), std::copysign(std::numeric_limits<T>::quiet_NaN(), T(1))));
            EXPECT_TRUE(same_bits<T>(Math::copysign(T(1), std::numeric_limits<T>::denorm_min()), std::copysign(T(1), std::numeric_limits<T>::denorm_min())));
            EXPECT_TRUE(same_bits<T>(Math::copysign(-T(0), T(-3)), std::copysign(-T(0), T(-3))));
            EXPECT_TRUE(same_bits<T>(Math::copysign(T(1), -T(1)), std::copysign(T(1), -T(1))));
        }

        TYPED_TEST(Small_DeviceMath, PowMatchesStdBitForBit)
        {
            using T = TypeParam;
            EXPECT_TRUE(same_bits<T>(Math::pow(T(7), T(3)), std::pow(T(7), T(3))));
            EXPECT_TRUE(same_bits<T>(Math::pow(T(-7), T(3)), std::pow(T(-7), T(3))));
            EXPECT_TRUE(same_bits<T>(Math::pow(T(0.5), T(-0.0)), std::pow(T(0.5), T(-0.0))));
            EXPECT_TRUE(same_bits<T>(Math::pow(T(-2.5), T(0)), std::pow(T(-2.5), T(0))));
            EXPECT_TRUE(same_bits<T>(Math::pow(std::numeric_limits<T>::max(), T(0.5)), std::pow(std::numeric_limits<T>::max(), T(0.5))));
            EXPECT_TRUE(same_bits<T>(Math::pow(std::numeric_limits<T>::infinity(), T(2)), std::pow(std::numeric_limits<T>::infinity(), T(2))));
            EXPECT_TRUE(same_bits<T>(Math::pow(std::numeric_limits<T>::quiet_NaN(), T(1)), std::pow(std::numeric_limits<T>::quiet_NaN(), T(1))));
            EXPECT_TRUE(same_bits<T>(Math::pow(T(1), std::numeric_limits<T>::denorm_min()), std::pow(T(1), std::numeric_limits<T>::denorm_min())));
            EXPECT_TRUE(same_bits<T>(Math::pow(-T(0), T(-3)), std::pow(-T(0), T(-3))));
            EXPECT_TRUE(same_bits<T>(Math::pow(T(1), -T(1)), std::pow(T(1), -T(1))));
        }

        TYPED_TEST(Small_DeviceMath, Atan2MatchesStdBitForBit)
        {
            using T = TypeParam;
            EXPECT_TRUE(same_bits<T>(Math::atan2(T(7), T(3)), std::atan2(T(7), T(3))));
            EXPECT_TRUE(same_bits<T>(Math::atan2(T(-7), T(3)), std::atan2(T(-7), T(3))));
            EXPECT_TRUE(same_bits<T>(Math::atan2(T(0.5), T(-0.0)), std::atan2(T(0.5), T(-0.0))));
            EXPECT_TRUE(same_bits<T>(Math::atan2(T(-2.5), T(0)), std::atan2(T(-2.5), T(0))));
            EXPECT_TRUE(same_bits<T>(Math::atan2(std::numeric_limits<T>::max(), T(0.5)), std::atan2(std::numeric_limits<T>::max(), T(0.5))));
            EXPECT_TRUE(same_bits<T>(Math::atan2(std::numeric_limits<T>::infinity(), T(2)), std::atan2(std::numeric_limits<T>::infinity(), T(2))));
            EXPECT_TRUE(same_bits<T>(Math::atan2(std::numeric_limits<T>::quiet_NaN(), T(1)), std::atan2(std::numeric_limits<T>::quiet_NaN(), T(1))));
            EXPECT_TRUE(same_bits<T>(Math::atan2(T(1), std::numeric_limits<T>::denorm_min()), std::atan2(T(1), std::numeric_limits<T>::denorm_min())));
            EXPECT_TRUE(same_bits<T>(Math::atan2(-T(0), T(-3)), std::atan2(-T(0), T(-3))));
            EXPECT_TRUE(same_bits<T>(Math::atan2(T(1), -T(1)), std::atan2(T(1), -T(1))));
        }

        /// @brief Names float, so a test reads the type of a result from the overload it selects.
        static const char* type_name(float)
        {
            return "float";
        }

        /// @brief Names double.
        static const char* type_name(double)
        {
            return "double";
        }

        /// @brief Refuses every other type, so a converted result cannot pass for the exact one.
        template<typename T>
        const char* type_name(T) = delete;

        /// @brief One unary host function, one input and the value the standard library gives.
        struct UnaryRow
        {
            const char* name;
            double (*function)(double);
            double input;
            double expected;
            double bound;
        };

        /// @brief One binary host function, two inputs and the value the standard library gives.
        struct BinaryRow
        {
            const char* name;
            double (*function)(double, double);
            double left;
            double right;
            double expected;
            double bound;
        };

        /**
         * @brief Expects a host result within bound of the expected value; a zero bound demands
         *        the exact value.
         */
        static void expect_host_result(double got, double expected, double bound)
        {
            EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(got, expected, bound, 1.0)))
                << "got " << got << ", expected " << expected;
        }

        /// @brief Names a row by its function, which gtest takes as the test's suffix.
        template<typename Row>
        static std::string row_name(const ::testing::TestParamInfo<Row>& info)
        {
            return info.param.name;
        }

        /// @brief Returns the unary rows; the literals are the correctly rounded values.
        static std::vector<UnaryRow> unary_rows()
        {
            return {
                {"abs", [](double x) { return Math::abs(x); }, -2.5, 2.5, 0.0},
                {"floor", [](double x) { return Math::floor(x); }, -1.5, -2.0, 0.0},
                {"sqrt", [](double x) { return Math::sqrt(x); }, 16.0, 4.0, 0.0},
                {"cbrt", [](double x) { return Math::cbrt(x); }, 27.0, 3.0, K_TOLERANCE},
                {"exp", [](double x) { return Math::exp(x); }, 1.0, 2.718281828459045, K_TOLERANCE},
                {"log", [](double x) { return Math::log(x); }, 2.718281828459045, 1.0, K_TOLERANCE},
                {"sin", [](double x) { return Math::sin(x); }, 0.5, 0.479425538604203, K_TOLERANCE},
                {"cos", [](double x) { return Math::cos(x); }, 0.5, 0.8775825618903728, K_TOLERANCE},
                {"asin", [](double x) { return Math::asin(x); }, 1.0, 1.5707963267948966, K_TOLERANCE},
            };
        }

        /// @brief Returns the binary rows.
        static std::vector<BinaryRow> binary_rows()
        {
            return {
                {"copysign", [](double x, double y) { return Math::copysign(x, y); }, 3.0, -0.0, -3.0, 0.0},
                {"fmod", [](double x, double y) { return Math::fmod(x, y); }, 7.0, 3.0, 1.0, 0.0},
                {"pow", [](double x, double y) { return Math::pow(x, y); }, 2.0, 10.0, 1024.0, K_TOLERANCE},
                {"atan2", [](double x, double y) { return Math::atan2(x, y); }, 1.0, -1.0, 2.356194490192345, K_TOLERANCE},
            };
        }

        /// @brief Runs one unary row of the host functions.
        class Small_DeviceMathUnary : public ::testing::TestWithParam<UnaryRow>
        {
        };

        TEST_P(Small_DeviceMathUnary, TheHostFunctionGivesTheStandardResult)
        {
            const UnaryRow& row = GetParam();
            expect_host_result(row.function(row.input), row.expected, row.bound);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_DeviceMathUnary, ::testing::ValuesIn(unary_rows()),
                                 row_name<UnaryRow>);

        /// @brief Runs one binary row of the host functions.
        class Small_DeviceMathBinary : public ::testing::TestWithParam<BinaryRow>
        {
        };

        TEST_P(Small_DeviceMathBinary, TheHostFunctionGivesTheStandardResult)
        {
            const BinaryRow& row = GetParam();
            expect_host_result(row.function(row.left, row.right), row.expected, row.bound);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_DeviceMathBinary, ::testing::ValuesIn(binary_rows()),
                                 row_name<BinaryRow>);

        TEST(Small_DeviceMath, IsFiniteSeparatesFiniteFromInfinityAndNaN)
        {
            EXPECT_TRUE(Math::isfinite(1.0f));
            EXPECT_FALSE(Math::isfinite(std::numeric_limits<float>::infinity()));
            EXPECT_FALSE(Math::isfinite(std::numeric_limits<double>::quiet_NaN()));
        }

        TEST(Small_DeviceMath, AbsKeepsTheTypeOfItsArgument)
        {
            // A double result would pass a value comparison, so the type is read from the overload.
            EXPECT_STREQ(type_name(Math::abs(-3.0f)), "float");
            EXPECT_STREQ(type_name(Math::abs(-3.0)), "double");
            EXPECT_EQ(Math::abs(-3.0f), 3.0f);
            EXPECT_EQ(Math::abs(-0.0), 0.0);
        }
    } // namespace Test
} // namespace SushiBLAS
