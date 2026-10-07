/**************************************************************************/
/* test_random_large_lambda.cpp                                           */
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
 * @file test_random_large_lambda.cpp
 * @brief Tests the Poisson draw for lambda in the PTRS range and its reproducibility.
 * @author Mustafa Garip
 */

#include <cmath>
#include <vector>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;
using SushiBLAS::Core::DataType;

namespace
{
    /// @brief Number of draws per moment test, 2^16.
    constexpr int64_t K_DRAWS = 65536;
    /// @brief Standard errors allowed between a sample moment and lambda.
    constexpr double K_STANDARD_ERRORS = 4.0;
    /// @brief Seed of the moment tests.
    constexpr uint64_t K_MOMENT_SEED = 20261006;
    /// @brief Seed of the reproducibility tests.
    constexpr uint64_t K_REPLAY_SEED = 7;

    /// @brief A mean in the PTRS range, near its lower end.
    constexpr double K_LAMBDA_SMALL = 50.0;
    /// @brief sqrt(lambda / N) for lambda 50: the standard error of the sample mean.
    constexpr double K_MEAN_SE_SMALL = 0.027621;
    /// @brief sqrt((2 lambda^2 + lambda) / N) for lambda 50: the standard error of the sample variance.
    constexpr double K_VARIANCE_SE_SMALL = 0.277591;

    /// @brief A mean in the middle of the PTRS range.
    constexpr double K_LAMBDA_MEDIUM = 5000.0;
    /// @brief sqrt(lambda / N) for lambda 5000.
    constexpr double K_MEAN_SE_MEDIUM = 0.276214;
    /// @brief sqrt((2 lambda^2 + lambda) / N) for lambda 5000.
    constexpr double K_VARIANCE_SE_MEDIUM = 873.51;

    /// @brief A mean near the top of the accepted range.
    constexpr double K_LAMBDA_LARGE = 1.0e6;
    /// @brief sqrt(lambda / N) for lambda 1e6, exactly 1000 / 256.
    constexpr double K_MEAN_SE_LARGE = 3.90625;
    /// @brief sqrt((2 lambda^2 + lambda) / N) for lambda 1e6.
    constexpr double K_VARIANCE_SE_LARGE = 5524.27;

    /// @brief A mean in the Knuth range, below the PTRS threshold of 10.
    constexpr double K_LAMBDA_KNUTH = 3.0;
}

class Unit_RandomLargeLambdaTest : public SushiBLASTest
{
    protected:
        /** @brief Returns K_DRAWS FLOAT64 Poisson draws of mean lambda from seed. */
        std::vector<double> draw(double lambda, uint64_t seed)
        {
            auto t = engine->create_tensor({K_DRAWS}, DataType::Float64);
            SushiBLAS::RandomOps(*engine).set_seed(seed);
            SushiBLAS::RandomOps(*engine).poisson(t, lambda);
            engine->execute();

            std::vector<double> host(static_cast<size_t>(K_DRAWS));
            auto& q = TestEnvironment::get_context().get_queue();
            q.memcpy(host.data(), t.data(), host.size() * sizeof(double)).wait();
            return host;
        }

        /** @brief Asserts non-negative integral draws whose mean and variance lie near lambda. */
        void expect_poisson_moments(double lambda, double mean_se, double variance_se)
        {
            const std::vector<double> values = draw(lambda, K_MOMENT_SEED);

            double sum = 0.0;
            for (double v : values)
            {
                ASSERT_GE(v, 0.0) << "A Poisson draw is negative.";
                ASSERT_EQ(v, std::floor(v)) << "A Poisson draw is not an integer.";
                sum += v;
            }
            const double mean = sum / static_cast<double>(K_DRAWS);

            double squares = 0.0;
            for (double v : values)
                squares += (v - mean) * (v - mean);
            const double variance = squares / static_cast<double>(K_DRAWS - 1);

            EXPECT_NEAR(mean, lambda, K_STANDARD_ERRORS * mean_se);
            EXPECT_NEAR(variance, lambda, K_STANDARD_ERRORS * variance_se);
        }

        /** @brief Asserts that two draws from one seed are the same bits. */
        void expect_replayed(double lambda)
        {
            const std::vector<double> first = draw(lambda, K_REPLAY_SEED);
            const std::vector<double> second = draw(lambda, K_REPLAY_SEED);
            EXPECT_EQ(0, std::memcmp(first.data(), second.data(), first.size() * sizeof(double)));
        }
};

TEST_F(Unit_RandomLargeLambdaTest, PoissonMomentsMatchLambdaFifty)
{
    expect_poisson_moments(K_LAMBDA_SMALL, K_MEAN_SE_SMALL, K_VARIANCE_SE_SMALL);
}

TEST_F(Unit_RandomLargeLambdaTest, PoissonMomentsMatchLambdaFiveThousand)
{
    expect_poisson_moments(K_LAMBDA_MEDIUM, K_MEAN_SE_MEDIUM, K_VARIANCE_SE_MEDIUM);
}

TEST_F(Unit_RandomLargeLambdaTest, PoissonMomentsMatchLambdaOneMillion)
{
    expect_poisson_moments(K_LAMBDA_LARGE, K_MEAN_SE_LARGE, K_VARIANCE_SE_LARGE);
}

TEST_F(Unit_RandomLargeLambdaTest, PoissonReplaysBitForBitInBothMethods)
{
    expect_replayed(K_LAMBDA_KNUTH);
    expect_replayed(K_LAMBDA_MEDIUM);
}

TEST_F(Unit_RandomLargeLambdaTest, PoissonAcceptsTheCap)
{
    auto t = engine->create_tensor({4}, DataType::Float64);
    EXPECT_NO_THROW(SushiBLAS::RandomOps(*engine).poisson(t, SushiBLAS::K_MAX_POISSON_LAMBDA));
    engine->execute();

    std::vector<double> host(4);
    auto& q = TestEnvironment::get_context().get_queue();
    q.memcpy(host.data(), t.data(), host.size() * sizeof(double)).wait();
    for (double v : host)
    {
        EXPECT_GE(v, 0.0);
        EXPECT_EQ(v, std::floor(v));
    }
}
