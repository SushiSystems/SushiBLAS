/**************************************************************************/
/* test_max_min_nan.cpp                                                   */
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
 * @file test_max_min_nan.cpp
 * @brief Tests reductions().max and min on NaN, infinite and signed-zero input.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

namespace
{
    /// @brief Holds a quiet NaN.
    const float K_NAN = std::numeric_limits<float>::quiet_NaN();

    /// @brief Holds positive infinity.
    const float K_INF = std::numeric_limits<float>::infinity();

    /// @brief Holds the element count of the large case, past the CPU two-stage threshold.
    constexpr std::size_t K_LARGE_SIZE = 100003;

    /// @brief Holds how far from the end the large case places its NaN.
    constexpr std::size_t K_NAN_OFFSET_FROM_END = 5;

    /// @brief Holds the absolute tolerance the finite expectations use.
    constexpr double K_TOLERANCE = 1e-6;
}

class Unit_MaxMinNaNTest : public SushiBLASTest
{
    protected:
        /**
         * @brief Reduces values with max and min and checks both results.
         */
        void expect_max_min(const std::vector<float>& values, float expected_max, float expected_min)
        {
            auto t = engine->create_tensor({static_cast<int64_t>(values.size())});
            fill_tensor(t, values);

            auto max = engine->create_tensor({1});
            auto min = engine->create_tensor({1});
            SushiBLAS::ReductionOps(*engine).max(t, max);
            SushiBLAS::ReductionOps(*engine).min(t, min);
            engine->execute();

            verify_tensor(max, {expected_max}, K_TOLERANCE);
            verify_tensor(min, {expected_min}, K_TOLERANCE);
        }
};

TEST_F(Unit_MaxMinNaNTest, NegativeInfinityAloneGivesNegativeInfinity)
{
    expect_max_min({-K_INF}, -K_INF, -K_INF);
}

TEST_F(Unit_MaxMinNaNTest, TwoNegativeInfinitiesGiveNegativeInfinity)
{
    expect_max_min({-K_INF, -K_INF}, -K_INF, -K_INF);
}

TEST_F(Unit_MaxMinNaNTest, PositiveInfinityAloneGivesPositiveInfinity)
{
    expect_max_min({K_INF}, K_INF, K_INF);
}

TEST_F(Unit_MaxMinNaNTest, SignedZerosGivePositiveMaxAndNegativeMinInEitherOrder)
{
    const std::vector<std::vector<float>> orders = {{-0.0f, 0.0f}, {0.0f, -0.0f}};

    for (const std::vector<float>& values : orders)
    {
        auto t = engine->create_tensor({2});
        fill_tensor(t, values);

        auto max = engine->create_tensor({1});
        auto min = engine->create_tensor({1});
        SushiBLAS::ReductionOps(*engine).max(t, max);
        SushiBLAS::ReductionOps(*engine).min(t, min);
        engine->execute();

        const float got_max = read_tensor<float>(max)[0];
        const float got_min = read_tensor<float>(min)[0];
        EXPECT_EQ(got_max, 0.0f);
        EXPECT_FALSE(std::signbit(got_max)) << "max of {-0, +0} must be +0";
        EXPECT_EQ(got_min, 0.0f);
        EXPECT_TRUE(std::signbit(got_min)) << "min of {-0, +0} must be -0";
    }
}

TEST_F(Unit_MaxMinNaNTest, MaxAlongARowOfNegativeInfinitiesGivesNegativeInfinity)
{
    auto x = engine->create_tensor({1, 2});
    fill_tensor(x, {-K_INF, -K_INF});

    auto out = engine->create_tensor({1});
    SushiBLAS::ReductionOps(*engine).max_along(x, 1, out);
    engine->execute();

    verify_tensor(out, {-K_INF}, K_TOLERANCE);
}

TEST_F(Unit_MaxMinNaNTest, NaNFirstGivesNaN)
{
    expect_max_min({K_NAN, 1.0f, 3.0f}, K_NAN, K_NAN);
}

TEST_F(Unit_MaxMinNaNTest, NaNInTheMiddleGivesNaN)
{
    expect_max_min({1.0f, K_NAN, 3.0f}, K_NAN, K_NAN);
}

TEST_F(Unit_MaxMinNaNTest, NaNLastGivesNaN)
{
    expect_max_min({1.0f, 3.0f, K_NAN}, K_NAN, K_NAN);
}

TEST_F(Unit_MaxMinNaNTest, WithoutNaNGivesTheExtremes)
{
    expect_max_min({1.0f, -2.0f, 3.0f}, 3.0f, -2.0f);
}

TEST_F(Unit_MaxMinNaNTest, NaNNearTheEndOfALargeInputGivesNaN)
{
    std::vector<float> values(K_LARGE_SIZE, 1.0f);
    values[K_LARGE_SIZE - K_NAN_OFFSET_FROM_END] = K_NAN;
    expect_max_min(values, K_NAN, K_NAN);
}

TEST_F(Unit_MaxMinNaNTest, LargeInputWithoutNaNGivesTheExtremes)
{
    // Ones everywhere, 7 at the front and -4 at the back: max 7, min -4.
    std::vector<float> values(K_LARGE_SIZE, 1.0f);
    values.front() = 7.0f;
    values.back() = -4.0f;
    expect_max_min(values, 7.0f, -4.0f);
}
