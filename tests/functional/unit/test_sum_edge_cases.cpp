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
 * @file test_sum_edge_cases.cpp
 * @brief Tests ReductionOps::sum, mean, max and min on edge-case inputs.
 * @author Mustafa Garip
 */

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
    /// @brief Holds positive infinity.
    const float K_INF = std::numeric_limits<float>::infinity();

    /// @brief Holds a quiet NaN.
    const float K_NAN = std::numeric_limits<float>::quiet_NaN();

    /// @brief Holds 2^16 + 1, one element past a power-of-two work size.
    constexpr size_t K_LARGE_SIZE = 65537;

    /// @brief Holds the absolute tolerance every finite expectation uses.
    constexpr double K_TOLERANCE = 1e-5;

    /// @brief Holds the four results one input is expected to reduce to.
    struct Expected
    {
        float sum;  /**< Expected sum. */
        float mean; /**< Expected mean. */
        float max;  /**< Expected maximum. */
        float min;  /**< Expected minimum. */
    };
}

class Unit_SumEdgeCasesTest : public SushiBLASTest
{
    protected:
        /**
         * @brief Reduces values with sum, mean, max and min and checks each result.
         */
        void expect_reductions(const std::vector<float>& values, const Expected& expected)
        {
            auto t = engine->create_tensor({static_cast<int64_t>(values.size())});
            fill_tensor(t, values);

            auto sum = engine->create_tensor({1});
            auto mean = engine->create_tensor({1});
            auto max = engine->create_tensor({1});
            auto min = engine->create_tensor({1});
            SushiBLAS::ReductionOps(*engine).sum(t, sum);
            SushiBLAS::ReductionOps(*engine).mean(t, mean);
            SushiBLAS::ReductionOps(*engine).max(t, max);
            SushiBLAS::ReductionOps(*engine).min(t, min);
            engine->execute();

            verify_tensor(sum, {expected.sum}, K_TOLERANCE);
            verify_tensor(mean, {expected.mean}, K_TOLERANCE);
            verify_tensor(max, {expected.max}, K_TOLERANCE);
            verify_tensor(min, {expected.min}, K_TOLERANCE);
        }

        /**
         * @brief Reduces values with sum and mean only and checks both results.
         */
        void expect_sum_and_mean(const std::vector<float>& values, float expected_sum, float expected_mean)
        {
            auto t = engine->create_tensor({static_cast<int64_t>(values.size())});
            fill_tensor(t, values);

            auto sum = engine->create_tensor({1});
            auto mean = engine->create_tensor({1});
            SushiBLAS::ReductionOps(*engine).sum(t, sum);
            SushiBLAS::ReductionOps(*engine).mean(t, mean);
            engine->execute();

            verify_tensor(sum, {expected_sum}, K_TOLERANCE);
            verify_tensor(mean, {expected_mean}, K_TOLERANCE);
        }
};

TEST_F(Unit_SumEdgeCasesTest, SingleElement)
{
    expect_reductions({-4.25f}, {-4.25f, -4.25f, -4.25f, -4.25f});
}

TEST_F(Unit_SumEdgeCasesTest, OnePastAPowerOfTwo)
{
    // 65535 ones, -1 first and 3 last: sum 65535 - 1 + 3 = 65537, mean 65537 / 65537 = 1.
    std::vector<float> values(K_LARGE_SIZE, 1.0f);
    values.front() = -1.0f;
    values.back() = 3.0f;
    expect_reductions(values, {65537.0f, 1.0f, 3.0f, -1.0f});
}

TEST_F(Unit_SumEdgeCasesTest, AllEqual)
{
    // Seven elements of 2.5: sum 17.5.
    expect_reductions(std::vector<float>(7, 2.5f), {17.5f, 2.5f, 2.5f, 2.5f});
}

TEST_F(Unit_SumEdgeCasesTest, NegativeZeros)
{
    expect_reductions({-0.0f, -0.0f, -0.0f}, {0.0f, 0.0f, 0.0f, 0.0f});
}

TEST_F(Unit_SumEdgeCasesTest, NaNPropagatesThroughSumAndMean)
{
    expect_sum_and_mean({1.0f, K_NAN, 2.0f}, K_NAN, K_NAN);
}

TEST_F(Unit_SumEdgeCasesTest, NaNPropagatesThroughEveryReduction)
{
    expect_reductions({1.0f, K_NAN, 2.0f}, {K_NAN, K_NAN, K_NAN, K_NAN});
}

TEST_F(Unit_SumEdgeCasesTest, PositiveInfinity)
{
    expect_reductions({1.0f, K_INF, -2.0f}, {K_INF, K_INF, K_INF, -2.0f});
}

TEST_F(Unit_SumEdgeCasesTest, NegativeInfinity)
{
    expect_reductions({1.0f, -K_INF, -2.0f}, {-K_INF, -K_INF, 1.0f, -K_INF});
}
