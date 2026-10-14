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
 * @brief Tests max, min, max_along and min_along on NaN, infinite and signed-zero input.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/reduction/reduce_axis.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include "test_utils.hpp"

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

    /// @brief Holds the row count of the axis case.
    constexpr std::size_t K_AXIS_ROWS = 3;

    /// @brief Holds the column count of the axis case, a multiple of no work-group width.
    constexpr std::size_t K_AXIS_COLUMNS = 733;

    /// @brief Holds a value no row of the axis case reduces to.
    constexpr float K_UNWRITTEN = 99.0f;
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
            engine->execute().value();

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
        engine->execute().value();

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
    engine->execute().value();

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

/**
 * @brief Tests that a NaN makes its own row's extrema NaN in both launch shapes of the axis fold.
 */
TEST_F(Unit_MaxMinNaNTest, AxisExtremaPropagateNaN)
{
    namespace Reduce = SushiBLAS::Reduce;

    // Row r holds r everywhere, r + 2 at column 5 and r - 2 at column 300; row 1 holds a NaN.
    std::vector<float> values(K_AXIS_ROWS * K_AXIS_COLUMNS);
    for (std::size_t r = 0; r < K_AXIS_ROWS; ++r)
    {
        for (std::size_t c = 0; c < K_AXIS_COLUMNS; ++c)
            values[r * K_AXIS_COLUMNS + c] = static_cast<float>(r);
        values[r * K_AXIS_COLUMNS + 5] = static_cast<float>(r) + 2.0f;
        values[r * K_AXIS_COLUMNS + 300] = static_cast<float>(r) - 2.0f;
    }
    values[K_AXIS_COLUMNS + 421] = K_NAN;

    const std::vector<float> expected_max = {2.0f, K_NAN, 4.0f};
    const std::vector<float> expected_min = {-2.0f, K_NAN, 0.0f};

    auto x = engine->create_tensor({static_cast<int64_t>(K_AXIS_ROWS),
                                    static_cast<int64_t>(K_AXIS_COLUMNS)});
    fill_tensor(x, values);

    auto max = engine->create_tensor({static_cast<int64_t>(K_AXIS_ROWS)});
    auto min = engine->create_tensor({static_cast<int64_t>(K_AXIS_ROWS)});
    SushiBLAS::ReductionOps(*engine).max_along(x, 1, max);
    SushiBLAS::ReductionOps(*engine).min_along(x, 1, min);
    engine->execute().value();

    verify_tensor(max, expected_max, K_TOLERANCE);
    verify_tensor(min, expected_min, K_TOLERANCE);

    auto& q = TestEnvironment::queue();
    const std::size_t local = Reduce::group_width(SushiBLAS::Device::profile_for(q));
    ASSERT_GT(local, 1u) << "the cooperative shape needs a group wider than one";

    Reduce::AxisSpan span;
    span.segments = K_AXIS_ROWS;
    span.extent = K_AXIS_COLUMNS;
    span.segment_stride = K_AXIS_COLUMNS;
    span.element_stride = 1;
    span.out_stride = 1;

    const auto identity = [](float value) { return value; };
    const auto keep = [](float value, [[maybe_unused]] std::size_t n) { return value; };

    for (std::size_t width : {std::size_t{1}, local})
    {
        SCOPED_TRACE(width == 1 ? "item-per-segment shape" : "work-group tree");

        auto shape_max = engine->create_tensor({static_cast<int64_t>(K_AXIS_ROWS)});
        auto shape_min = engine->create_tensor({static_cast<int64_t>(K_AXIS_ROWS)});
        fill_tensor(shape_max, std::vector<float>(K_AXIS_ROWS, K_UNWRITTEN));
        fill_tensor(shape_min, std::vector<float>(K_AXIS_ROWS, K_UNWRITTEN));

        Reduce::reduce_axis<Reduce::NaNPropagatingMax>(
            q, x.data_as<float>(), shape_max.data_as<float>(), span,
            Reduce::AxisPlan{width}, identity, keep, {}).wait();
        Reduce::reduce_axis<Reduce::NaNPropagatingMin>(
            q, x.data_as<float>(), shape_min.data_as<float>(), span,
            Reduce::AxisPlan{width}, identity, keep, {}).wait();

        verify_tensor(shape_max, expected_max, K_TOLERANCE);
        verify_tensor(shape_min, expected_min, K_TOLERANCE);
    }
}
