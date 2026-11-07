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
 * @file test_max_min_nan.cpp
 * @brief Tests max, min, max_along and min_along on NaN, infinite and signed-zero input.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/reduction/reduce_axis.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
using namespace SushiBLAS::Test;

namespace
{
    namespace Reduce = SushiBLAS::Reduce;

    /// @brief Holds a quiet NaN.
    const float K_NAN = std::numeric_limits<float>::quiet_NaN();

    /// @brief Holds positive infinity.
    const float K_INF = std::numeric_limits<float>::infinity();

    /// @brief Holds the element count of the large case, past the CPU two-stage threshold.
    constexpr std::size_t K_LARGE_SIZE = 100003;

    /// @brief Holds how far from the end the large case places its NaN.
    constexpr std::size_t K_NAN_OFFSET_FROM_END = 5;

    /// @brief Holds the row count of the axis case.
    constexpr std::size_t K_AXIS_ROWS = 3;

    /// @brief Holds the column count of the axis case, a multiple of no work-group width.
    constexpr std::size_t K_AXIS_COLUMNS = 733;

    /// @brief Holds a value no row of the axis case reduces to.
    constexpr float K_UNWRITTEN = 99.0f;

    /// @brief Returns K_LARGE_SIZE ones with a NaN K_NAN_OFFSET_FROM_END from the end.
    std::vector<float> large_with_nan_near_the_end()
    {
        std::vector<float> values(K_LARGE_SIZE, 1.0f);
        values[K_LARGE_SIZE - K_NAN_OFFSET_FROM_END] = K_NAN;
        return values;
    }

    /// @brief Returns K_LARGE_SIZE ones with 7 at the front and -4 at the back.
    std::vector<float> large_without_nan()
    {
        std::vector<float> values(K_LARGE_SIZE, 1.0f);
        values.front() = 7.0f;
        values.back() = -4.0f;
        return values;
    }

    /// @brief Returns a row-major rows x columns matrix whose row 1 holds a NaN.
    ///
    /// Row r holds r everywhere, r + 2 at column 5 and r - 2 at column 300.
    std::vector<float> rows_with_a_nan_in_row_one()
    {
        std::vector<float> values(K_AXIS_ROWS * K_AXIS_COLUMNS);
        for (std::size_t r = 0; r < K_AXIS_ROWS; ++r)
        {
            for (std::size_t c = 0; c < K_AXIS_COLUMNS; ++c)
                values[r * K_AXIS_COLUMNS + c] = static_cast<float>(r);
            values[r * K_AXIS_COLUMNS + 5] = static_cast<float>(r) + 2.0f;
            values[r * K_AXIS_COLUMNS + 300] = static_cast<float>(r) - 2.0f;
        }
        values[K_AXIS_COLUMNS + 421] = K_NAN;
        return values;
    }

    /// @brief Returns the maximum of each row of rows_with_a_nan_in_row_one().
    std::vector<float> axis_expected_max()
    {
        return {2.0f, K_NAN, 4.0f};
    }

    /// @brief Returns the minimum of each row of rows_with_a_nan_in_row_one().
    std::vector<float> axis_expected_min()
    {
        return {-2.0f, K_NAN, 0.0f};
    }
}

class MaxMinFixture : public SushiBLASTest
{
    protected:
        /**
         * @brief Reduces values with max and min and expects both results exactly.
         */
        void expect_max_min(const std::vector<float>& values, float expected_max, float expected_min)
        {
            auto t = engine->create_tensor({static_cast<int64_t>(values.size())});
            fill_tensor(t, values);

            auto max = engine->create_tensor({1});
            auto min = engine->create_tensor({1});
            SushiBLAS::ReductionOps(recorder()).max(t, max);
            SushiBLAS::ReductionOps(recorder()).min(t, min);
            recording->run().value();

            expect_tensor_equal(max, {expected_max});
            expect_tensor_equal(min, {expected_min});
        }

        /** @brief Returns the rows x columns input of the axis case as a filled tensor. */
        SushiBLAS::Tensor axis_input()
        {
            auto x = engine->create_tensor({static_cast<int64_t>(K_AXIS_ROWS),
                                            static_cast<int64_t>(K_AXIS_COLUMNS)});
            fill_tensor(x, rows_with_a_nan_in_row_one());
            return x;
        }
};

class Medium_MaxMinNaNTest : public MaxMinFixture
{
};

/** @brief One input of max and min, with the maximum and the minimum it must give. */
struct ExtremaRow
{
    const char* name;
    std::vector<float> values;
    float expected_max;
    float expected_min;
};

/** @brief Returns the infinity, NaN and finite inputs, each with its extremes. */
static std::vector<ExtremaRow> extrema_rows()
{
    return {
        {"TwoNegativeInfinities", {-K_INF, -K_INF}, -K_INF, -K_INF},
        {"APositiveInfinityAlone", {K_INF}, K_INF, K_INF},
        {"ANaNFirst", {K_NAN, 1.0f, 3.0f}, K_NAN, K_NAN},
        {"ANaNInTheMiddle", {1.0f, K_NAN, 3.0f}, K_NAN, K_NAN},
        {"ANaNLast", {1.0f, 3.0f, K_NAN}, K_NAN, K_NAN},
        {"ANaNNearTheEndOfALargeInput", large_with_nan_near_the_end(), K_NAN, K_NAN},
        {"NoNaN", {1.0f, -2.0f, 3.0f}, 3.0f, -2.0f},
        // Ones everywhere, 7 at the front and -4 at the back.
        {"ALargeInputWithoutNaN", large_without_nan(), 7.0f, -4.0f},
    };
}

/** @brief Names a row by its description, which gtest takes as the test's suffix. */
template<typename Row>
static std::string row_name(const ::testing::TestParamInfo<Row>& info)
{
    return info.param.name;
}

/** @brief Runs max and min over one row of extrema_rows(). */
class Medium_MaxMinExtrema : public MaxMinFixture,
                             public ::testing::WithParamInterface<ExtremaRow>
{
};

/**
 * @brief Tests that max and min give the extremes of the row, NaN and infinities included.
 */
TEST_P(Medium_MaxMinExtrema, MaxAndMinGiveTheExtremesOfTheInput)
{
    const ExtremaRow& row = GetParam();

    expect_max_min(row.values, row.expected_max, row.expected_min);
}

INSTANTIATE_TEST_SUITE_P(, Medium_MaxMinExtrema, ::testing::ValuesIn(extrema_rows()),
                         row_name<ExtremaRow>);

/** @brief One order of a negative and a positive zero. */
struct ZeroOrderRow
{
    const char* name;
    std::vector<float> values;
};

/** @brief Returns both orders of -0 and +0. */
static std::vector<ZeroOrderRow> zero_order_rows()
{
    return {{"NegativeFirst", {-0.0f, 0.0f}}, {"PositiveFirst", {0.0f, -0.0f}}};
}

/** @brief Runs max and min over one order of signed zeros. */
class Medium_MaxMinSignedZeros : public MaxMinFixture,
                                 public ::testing::WithParamInterface<ZeroOrderRow>
{
};

/**
 * @brief Tests that the max of signed zeros is +0 and the min is -0, whichever comes first.
 */
TEST_P(Medium_MaxMinSignedZeros, GivePositiveMaxAndNegativeMin)
{
    auto t = engine->create_tensor({2});
    fill_tensor(t, GetParam().values);

    auto max = engine->create_tensor({1});
    auto min = engine->create_tensor({1});
    SushiBLAS::ReductionOps(recorder()).max(t, max);
    SushiBLAS::ReductionOps(recorder()).min(t, min);
    recording->run().value();

    const float got_max = read_tensor<float>(max)[0];
    const float got_min = read_tensor<float>(min)[0];
    EXPECT_EQ(got_max, 0.0f);
    EXPECT_FALSE(std::signbit(got_max)) << "max of {-0, +0} must be +0";
    EXPECT_EQ(got_min, 0.0f);
    EXPECT_TRUE(std::signbit(got_min)) << "min of {-0, +0} must be -0";
}

INSTANTIATE_TEST_SUITE_P(, Medium_MaxMinSignedZeros, ::testing::ValuesIn(zero_order_rows()),
                         row_name<ZeroOrderRow>);

TEST_F(Medium_MaxMinNaNTest, MaxAlongARowOfNegativeInfinitiesGivesNegativeInfinity)
{
    auto x = engine->create_tensor({1, 2});
    fill_tensor(x, {-K_INF, -K_INF});

    auto out = engine->create_tensor({1});
    SushiBLAS::ReductionOps(recorder()).max_along(x, 1, out);
    recording->run().value();

    expect_tensor_equal(out, {-K_INF});
}

/**
 * @brief Tests that a NaN makes its own row's maximum NaN and leaves the other rows alone.
 */
TEST_F(Medium_MaxMinNaNTest, MaxAlongPropagatesNaNOfItsOwnRow)
{
    auto x = axis_input();
    auto max = engine->create_tensor({static_cast<int64_t>(K_AXIS_ROWS)});

    SushiBLAS::ReductionOps(recorder()).max_along(x, 1, max);
    recording->run().value();

    expect_tensor_equal(max, axis_expected_max());
}

/**
 * @brief Tests that a NaN makes its own row's minimum NaN and leaves the other rows alone.
 */
TEST_F(Medium_MaxMinNaNTest, MinAlongPropagatesNaNOfItsOwnRow)
{
    auto x = axis_input();
    auto min = engine->create_tensor({static_cast<int64_t>(K_AXIS_ROWS)});

    SushiBLAS::ReductionOps(recorder()).min_along(x, 1, min);
    recording->run().value();

    expect_tensor_equal(min, axis_expected_min());
}

/** @brief Which of the two launch shapes of the axis fold a case drives. */
enum class AxisShape : std::uint8_t
{
    ItemPerSegment,
    WorkGroupTree,
};

/** @brief Names an axis shape for gtest. */
static std::string axis_shape_name(const ::testing::TestParamInfo<AxisShape>& info)
{
    return info.param == AxisShape::ItemPerSegment ? "ItemPerSegment" : "WorkGroupTree";
}

/** @brief Drives one launch shape of the axis fold directly. */
class Medium_MaxMinAxisShape : public MaxMinFixture,
                               public ::testing::WithParamInterface<AxisShape>
{
    protected:
        /** @brief Returns the group width of the shape under test. */
        std::size_t width()
        {
            const std::size_t local = Reduce::group_width(
                SushiBLAS::Device::profile_for(TestEnvironment::queue()));
            return GetParam() == AxisShape::ItemPerSegment ? std::size_t{1} : local;
        }

        /** @brief Returns the span of K_AXIS_ROWS contiguous rows of K_AXIS_COLUMNS elements. */
        static Reduce::AxisSpan row_span()
        {
            Reduce::AxisSpan span;
            span.segments = K_AXIS_ROWS;
            span.extent = K_AXIS_COLUMNS;
            span.segment_stride = K_AXIS_COLUMNS;
            span.element_stride = 1;
            span.out_stride = 1;
            return span;
        }

        /** @brief Returns a result tensor holding K_UNWRITTEN in every row. */
        SushiBLAS::Tensor unwritten_result()
        {
            auto out = engine->create_tensor({static_cast<int64_t>(K_AXIS_ROWS)});
            fill_tensor(out, std::vector<float>(K_AXIS_ROWS, K_UNWRITTEN));
            return out;
        }
};

/**
 * @brief Tests that the shape under test gives a NaN row a NaN maximum.
 */
TEST_P(Medium_MaxMinAxisShape, TheMaximumOfARowWithANaNIsNaN)
{
    auto x = axis_input();
    auto out = unwritten_result();

    const auto identity = [](float value) { return value; };
    const auto keep = [](float value, [[maybe_unused]] std::size_t n) { return value; };
    auto& q = TestEnvironment::queue();

    Reduce::reduce_axis<Reduce::NaNPropagatingMax>(
        q, x.data_as<float>(), out.data_as<float>(), row_span(), Reduce::AxisPlan{width()},
        identity, keep, {}).wait();

    expect_tensor_equal(out, axis_expected_max());
}

/**
 * @brief Tests that the shape under test gives a NaN row a NaN minimum.
 */
TEST_P(Medium_MaxMinAxisShape, TheMinimumOfARowWithANaNIsNaN)
{
    auto x = axis_input();
    auto out = unwritten_result();

    const auto identity = [](float value) { return value; };
    const auto keep = [](float value, [[maybe_unused]] std::size_t n) { return value; };
    auto& q = TestEnvironment::queue();

    Reduce::reduce_axis<Reduce::NaNPropagatingMin>(
        q, x.data_as<float>(), out.data_as<float>(), row_span(), Reduce::AxisPlan{width()},
        identity, keep, {}).wait();

    expect_tensor_equal(out, axis_expected_min());
}

INSTANTIATE_TEST_SUITE_P(, Medium_MaxMinAxisShape,
                         ::testing::Values(AxisShape::ItemPerSegment, AxisShape::WorkGroupTree),
                         axis_shape_name);
