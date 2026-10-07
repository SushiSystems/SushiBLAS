/**************************************************************************/
/* test_uniform.cpp                                                       */
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
 * @file test_uniform.cpp
 * @brief Tests RandomOps::uniform's bounds, sample moments, spread and seed determinism.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

namespace
{
    /// @brief Holds the number of draws each statistical test takes.
    constexpr int64_t K_DRAWS = 100000;

    /// @brief Holds the lower bound of the range.
    constexpr double K_MIN = -2.0;

    /// @brief Holds the upper bound of the range.
    constexpr double K_MAX = 6.0;

    /// @brief Holds the distribution's mean: (-2 + 6) / 2.
    constexpr double K_EXPECTED_MEAN = 2.0;

    /// @brief Holds the distribution's variance: 8^2 / 12.
    constexpr double K_EXPECTED_VARIANCE = 5.333333;

    /// @brief Holds four standard errors of the sample mean: 4 * sqrt(64 / 12) / sqrt(100000).
    constexpr double K_MEAN_BOUND = 0.029212;

    /// @brief Holds four standard errors of the sample variance: 4 * sqrt((8^4 / 80 - (64 / 12)^2) / 100000).
    constexpr double K_VARIANCE_BOUND = 0.060340;

    /// @brief Holds the band at each end the samples must reach: 1% of the range width.
    constexpr double K_EDGE_BAND = 0.08;

    /// @brief Holds the seed the determinism tests share.
    constexpr uint64_t K_SEED = 20261006;

    /// @brief Holds a seed different from K_SEED.
    constexpr uint64_t K_OTHER_SEED = 7;
}

class Unit_UniformTest : public SushiBLASTest
{
    protected:
        /**
         * @brief Draws K_DRAWS uniform samples of type T and returns them widened to double.
         * @tparam T float or double.
         */
        template <typename T>
        std::vector<double> draw(uint64_t seed)
        {
            auto t = engine->create_tensor({K_DRAWS}, get_sushiblas_dtype<T>());
            SushiBLAS::RandomOps(*engine).set_seed(seed);
            SushiBLAS::RandomOps(*engine).uniform(t, K_MIN, K_MAX);
            engine->execute();
            const std::vector<T> host = read_tensor<T>(t);
            return std::vector<double>(host.begin(), host.end());
        }

        /**
         * @brief Expects the samples to stay in [K_MIN, K_MAX), fill the range and match its moments.
         */
        static void expect_uniform_statistics(const std::vector<double>& samples)
        {
            double sum = 0.0;
            for (double v : samples)
            {
                ASSERT_GE(v, K_MIN);
                ASSERT_LT(v, K_MAX);
                sum += v;
            }
            const double mean = sum / static_cast<double>(samples.size());

            double squares = 0.0;
            for (double v : samples)
                squares += (v - mean) * (v - mean);
            const double variance = squares / static_cast<double>(samples.size() - 1);

            const auto [lowest, highest] = std::minmax_element(samples.begin(), samples.end());

            EXPECT_NEAR(mean, K_EXPECTED_MEAN, K_MEAN_BOUND);
            EXPECT_NEAR(variance, K_EXPECTED_VARIANCE, K_VARIANCE_BOUND);
            EXPECT_LT(*lowest, *highest) << "The fill is constant.";
            EXPECT_LT(*lowest, K_MIN + K_EDGE_BAND);
            EXPECT_GT(*highest, K_MAX - K_EDGE_BAND);
        }
};

TEST_F(Unit_UniformTest, Float32StaysInRangeAndMatchesItsMoments)
{
    expect_uniform_statistics(draw<float>(K_SEED));
}

TEST_F(Unit_UniformTest, Float64StaysInRangeAndMatchesItsMoments)
{
    expect_uniform_statistics(draw<double>(K_SEED));
}

TEST_F(Unit_UniformTest, EqualSeedsDrawEqualSamples)
{
    const std::vector<double> first = draw<float>(K_SEED);
    const std::vector<double> second = draw<float>(K_SEED);
    EXPECT_EQ(first, second);
}

TEST_F(Unit_UniformTest, DifferentSeedsDrawDifferentSamples)
{
    EXPECT_NE(draw<float>(K_SEED), draw<float>(K_OTHER_SEED));
}
