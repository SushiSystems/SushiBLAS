/**************************************************************************/
/* test_sum_edge_cases.cpp                                                */
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
 * @file test_sum_edge_cases.cpp
 * @brief Tests ReductionOps::sum, mean, max and min on edge-case inputs.
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
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
using namespace SushiBLAS::Test;

namespace
{
    /// @brief Holds positive infinity.
    const float K_INF = std::numeric_limits<float>::infinity();

    /// @brief Holds a quiet NaN.
    const float K_NAN = std::numeric_limits<float>::quiet_NaN();

    /// @brief Holds 2^16 + 1, one element past a power-of-two work size.
    constexpr size_t K_LARGE_SIZE = 65537;

    /// @brief Holds the four results one input is expected to reduce to.
    struct Expected
    {
        float sum;  /**< Expected sum. */
        float mean; /**< Expected mean. */
        float max;  /**< Expected maximum. */
        float min;  /**< Expected minimum. */
    };

    /// @brief Returns 65535 ones with -1 first and 3 last.
    std::vector<float> ones_between_minus_one_and_three()
    {
        std::vector<float> values(K_LARGE_SIZE, 1.0f);
        values.front() = -1.0f;
        values.back() = 3.0f;
        return values;
    }
}

class Medium_SumEdgeCasesTest : public SushiBLASTest
{
    protected:
        /**
         * @brief Reduces values with sum, mean, max and min and expects each result exactly.
         *
         * Every expectation is an integer or a dyadic fraction, a NaN or an infinity, so the
         * comparison is equality.
         */
        void expect_reductions(const std::vector<float>& values, const Expected& expected)
        {
            auto t = engine->create_tensor({static_cast<int64_t>(values.size())});
            fill_tensor(t, values);

            auto sum = engine->create_tensor({1});
            auto mean = engine->create_tensor({1});
            auto max = engine->create_tensor({1});
            auto min = engine->create_tensor({1});
            SushiBLAS::ReductionOps(recorder()).sum(t, sum);
            SushiBLAS::ReductionOps(recorder()).mean(t, mean);
            SushiBLAS::ReductionOps(recorder()).max(t, max);
            SushiBLAS::ReductionOps(recorder()).min(t, min);
            recording->run().value();

            expect_tensor_equal(sum, {expected.sum});
            expect_tensor_equal(mean, {expected.mean});
            expect_tensor_equal(max, {expected.max});
            expect_tensor_equal(min, {expected.min});
        }
};

/** @brief One input of the edge cases and the four results it reduces to. */
struct EdgeRow
{
    const char* name;
    std::vector<float> values;
    Expected expected;
};

/** @brief Returns the single element, equal elements, odd tail, zero, NaN and infinite inputs. */
static std::vector<EdgeRow> edge_rows()
{
    return {
        {"SingleElement", {-4.25f}, {-4.25f, -4.25f, -4.25f, -4.25f}},
        // Seven elements of 2.5: sum 17.5.
        {"AllEqual", std::vector<float>(7, 2.5f), {17.5f, 2.5f, 2.5f, 2.5f}},
        // 65535 ones, -1 first and 3 last: sum 65535 - 1 + 3 = 65537, mean 65537 / 65537 = 1.
        {"OnePastAPowerOfTwo", ones_between_minus_one_and_three(), {65537.0f, 1.0f, 3.0f, -1.0f}},
        // Equality cannot see the sign of zero; NegativeZerosKeepTheirSignInMaxAndMin does.
        {"NegativeZerosReduceToZero", {-0.0f, -0.0f, -0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}},
        {"ANaNPropagatesThroughEveryReduction", {1.0f, K_NAN, 2.0f}, {K_NAN, K_NAN, K_NAN, K_NAN}},
        {"APositiveInfinity", {1.0f, K_INF, -2.0f}, {K_INF, K_INF, K_INF, -2.0f}},
        {"ANegativeInfinity", {1.0f, -K_INF, -2.0f}, {-K_INF, -K_INF, 1.0f, -K_INF}},
    };
}

/** @brief Runs sum, mean, max and min over one row of edge_rows(). */
class Medium_SumEdgeCases : public Medium_SumEdgeCasesTest,
                            public ::testing::WithParamInterface<EdgeRow>
{
};

/**
 * @brief Tests that sum, mean, max and min give the expected results on an edge-case input.
 */
TEST_P(Medium_SumEdgeCases, SumMeanMaxAndMinGiveTheExpectedResults)
{
    expect_reductions(GetParam().values, GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(, Medium_SumEdgeCases, ::testing::ValuesIn(edge_rows()),
                         [](const ::testing::TestParamInfo<EdgeRow>& info)
                         {
                             return std::string(info.param.name);
                         });

/**
 * @brief Tests that the maximum and the minimum of negative zeros are negative zeros.
 */
TEST_F(Medium_SumEdgeCasesTest, NegativeZerosKeepTheirSignInMaxAndMin)
{
    auto t = engine->create_tensor({3});
    fill_tensor(t, {-0.0f, -0.0f, -0.0f});

    auto max = engine->create_tensor({1});
    auto min = engine->create_tensor({1});
    SushiBLAS::ReductionOps(recorder()).max(t, max);
    SushiBLAS::ReductionOps(recorder()).min(t, min);
    recording->run().value();

    const float got_max = read_tensor<float>(max)[0];
    const float got_min = read_tensor<float>(min)[0];
    EXPECT_EQ(got_max, 0.0f);
    EXPECT_TRUE(std::signbit(got_max)) << "max of {-0, -0, -0} must be -0";
    EXPECT_EQ(got_min, 0.0f);
    EXPECT_TRUE(std::signbit(got_min)) << "min of {-0, -0, -0} must be -0";
}
