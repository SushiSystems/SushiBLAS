/**************************************************************************/
/* test_normal.cpp                                                        */
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
 * @file test_normal.cpp
 * @brief Tests RandomOps::normal's sample moments, spread and seed determinism.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
using namespace SushiBLAS::Test;

namespace
{
    /// @brief Holds the number of draws each statistical test takes.
    constexpr int64_t K_DRAWS = 100000;

    /// @brief Holds the requested mean.
    constexpr double K_MEAN = 2.0;

    /// @brief Holds the requested standard deviation.
    constexpr double K_STDDEV = 3.0;

    /// @brief Holds four standard errors of the sample mean: 4 * 3 / sqrt(100000).
    constexpr double K_MEAN_BOUND = 0.037947;

    /// @brief Holds four standard errors of the sample variance: 4 * 9 * sqrt(2 / 100000).
    constexpr double K_VARIANCE_BOUND = 0.16100;

    /// @brief Holds the seed the determinism tests share.
    constexpr uint64_t K_SEED = 20261006;

    /// @brief Holds a seed different from K_SEED.
    constexpr uint64_t K_OTHER_SEED = 7;
}

class Unit_NormalTest : public SushiBLASTest
{
    protected:
        /**
         * @brief Draws K_DRAWS normal samples of type T and returns them widened to double.
         * @tparam T float or double.
         */
        template <typename T>
        std::vector<double> draw(uint64_t seed)
        {
            auto t = engine->create_tensor({K_DRAWS}, sushiblas_dtype<T>());
            SushiBLAS::RandomOps(*engine).set_seed(seed);
            SushiBLAS::RandomOps(*engine).normal(t, K_MEAN, K_STDDEV);
            engine->execute().value();
            const std::vector<T> host = read_tensor<T>(t);
            return std::vector<double>(host.begin(), host.end());
        }

        /**
         * @brief Expects the samples' mean, variance and spread to match N(K_MEAN, K_STDDEV^2).
         */
        static void expect_normal_statistics(const std::vector<double>& samples)
        {
            double sum = 0.0;
            for (double v : samples)
            {
                ASSERT_TRUE(std::isfinite(v));
                sum += v;
            }
            const double mean = sum / static_cast<double>(samples.size());

            double squares = 0.0;
            for (double v : samples)
                squares += (v - mean) * (v - mean);
            const double variance = squares / static_cast<double>(samples.size() - 1);

            const auto [lowest, highest] = std::minmax_element(samples.begin(), samples.end());

            EXPECT_NEAR(mean, K_MEAN, K_MEAN_BOUND);
            EXPECT_NEAR(variance, K_STDDEV * K_STDDEV, K_VARIANCE_BOUND);
            EXPECT_LT(*lowest, *highest) << "The fill is constant.";
            EXPECT_LT(*lowest, K_MEAN - 3.0 * K_STDDEV);
            EXPECT_GT(*highest, K_MEAN + 3.0 * K_STDDEV);
        }
};

TEST_F(Unit_NormalTest, Float32MomentsMatchTheRequestedDistribution)
{
    expect_normal_statistics(draw<float>(K_SEED));
}

TEST_F(Unit_NormalTest, Float64MomentsMatchTheRequestedDistribution)
{
    expect_normal_statistics(draw<double>(K_SEED));
}

TEST_F(Unit_NormalTest, EqualSeedsDrawEqualSamples)
{
    const std::vector<double> first = draw<float>(K_SEED);
    const std::vector<double> second = draw<float>(K_SEED);
    EXPECT_EQ(first, second);
}

TEST_F(Unit_NormalTest, DifferentSeedsDrawDifferentSamples)
{
    EXPECT_NE(draw<float>(K_SEED), draw<float>(K_OTHER_SEED));
}
