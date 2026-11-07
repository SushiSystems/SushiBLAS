/**************************************************************************/
/* test_distributions.cpp                                                 */
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
 * @file test_distributions.cpp
 * @brief Tests the log-normal, exponential, Poisson, Bernoulli, discrete and truncated draws.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>
#include "../random_statistics.hpp"
using namespace SushiBLAS::Test;
using SushiBLAS::Core::DataType;
using SushiBLAS::RandomOps;
using SushiBLAS::Tensor;

namespace
{
    /// @brief Holds the number of draws each statistical test takes.
    constexpr int64_t K_DRAWS = 65536;

    /// @brief Holds the number of draws of the bounds test.
    constexpr int64_t K_BOUNDED_DRAWS = 4096;

    /// @brief Holds the seed the statistical tests share.
    constexpr uint64_t K_SEED = 20261005;

    /// @brief Holds the seed of the far-lower-bound test.
    constexpr uint64_t K_FAR_BOUND_SEED = 20261007;

    /// @brief Holds the standard deviation of the underlying normal of the log-normal draw.
    constexpr double K_LOG_NORMAL_STDDEV = 0.5;

    /// @brief Holds the rate of the exponential draw.
    constexpr double K_EXPONENTIAL_RATE = 1.5;

    /// @brief Holds the probability of a 1 in the Bernoulli draw.
    constexpr double K_PROBABILITY = 0.3;

    /// @brief Holds the lowest integer of the discrete uniform draw.
    constexpr int64_t K_LOWEST_INTEGER = 0;

    /// @brief Holds the highest integer of the discrete uniform draw.
    constexpr int64_t K_HIGHEST_INTEGER = 10;

    /// @brief Holds a Poisson mean below 10, where the draw is the Knuth product.
    constexpr double K_KNUTH_LAMBDA = 4.0;

    /// @brief Holds the FLOAT32 roundings the exp of the mean may differ from the exact value by.
    constexpr std::size_t K_EXP_OPERATIONS = 8;

    /// @brief Holds a lower bound far below any draw; the kernel clamps it to -40.
    constexpr double K_FAR_LOWER_BOUND = -1.0e30;

    /** @brief Records one operation on a tensor. */
    using Record = std::function<void(RandomOps&, Tensor&)>;
}

class Small_DistributionsTest : public HostEngineTest
{
    protected:
        /** @brief Records one fill on count FLOAT32 elements from seed, runs it and returns the elements widened. */
        std::vector<double> draw(uint64_t seed, const Record& record, int64_t count = K_DRAWS)
        {
            auto t = engine->create_tensor({count});
            RandomOps random(recorder(), engine->rng());
            random.set_seed(seed);
            record(random, t);
            EXPECT_TRUE(recording->run_succeeds());
            return widened(read_tensor<float>(t));
        }

        /** @brief Expects every sample to lie inside [low, high]. */
        static void expect_inside(const std::vector<double>& samples, double low, double high)
        {
            const SampleMoments measured = measure_moments(samples);
            EXPECT_GE(measured.lowest, low);
            EXPECT_LE(measured.highest, high);
        }

        /** @brief Expects every sample to be a whole number. */
        static void expect_whole(const std::vector<double>& samples)
        {
            for (const double value : samples)
                ASSERT_EQ(value, std::floor(value));
        }
};

TEST_F(Small_DistributionsTest, LogNormalMomentsMatchItsLaw)
{
    const std::vector<double> samples = draw(K_SEED, [](RandomOps& random, Tensor& t)
    {
        random.log_normal(t, 0.0, K_LOG_NORMAL_STDDEV);
    });

    expect_moments_match(samples, log_normal_law(0.0, K_LOG_NORMAL_STDDEV));
}

TEST_F(Small_DistributionsTest, ExponentialMomentsMatchItsLaw)
{
    const std::vector<double> samples = draw(K_SEED, [](RandomOps& random, Tensor& t)
    {
        random.exponential(t, K_EXPONENTIAL_RATE);
    });

    expect_moments_match(samples, exponential_law(K_EXPONENTIAL_RATE));
}

TEST_F(Small_DistributionsTest, PoissonBelowTenMatchesItsMoments)
{
    const std::vector<double> samples = draw(K_SEED, [](RandomOps& random, Tensor& t)
    {
        random.poisson(t, K_KNUTH_LAMBDA);
    });

    expect_whole(samples);
    expect_moments_match(samples, poisson_law(K_KNUTH_LAMBDA));
}

TEST_F(Small_DistributionsTest, BernoulliMomentsMatchItsLaw)
{
    const std::vector<double> samples = draw(K_SEED, [](RandomOps& random, Tensor& t)
    {
        random.bernoulli(t, K_PROBABILITY);
    });

    expect_inside(samples, 0.0, 1.0);
    expect_moments_match(samples, bernoulli_law(K_PROBABILITY));
}

TEST_F(Small_DistributionsTest, DiscreteUniformMomentsMatchItsLaw)
{
    const std::vector<double> samples = draw(K_SEED, [](RandomOps& random, Tensor& t)
    {
        random.discrete_uniform(t, K_LOWEST_INTEGER, K_HIGHEST_INTEGER);
    });

    expect_whole(samples);
    expect_inside(samples, static_cast<double>(K_LOWEST_INTEGER),
                  static_cast<double>(K_HIGHEST_INTEGER));
    expect_moments_match(samples, discrete_uniform_law(static_cast<double>(K_LOWEST_INTEGER),
                                                       static_cast<double>(K_HIGHEST_INTEGER)));
}

/** @brief Names a truncated normal fill and the interval its samples must stay inside. */
struct BoundsRow
{
    const char* name; /**< The row name a failure shows. */
    double mean;      /**< The mean of the underlying normal. */
    double stddev;    /**< The standard deviation of the underlying normal. */
    double lower;     /**< The lower cut, in standard deviations. */
    double upper;     /**< The upper cut, in standard deviations. */
};

/** @brief Returns the fills of the bounds test: a shifted and scaled one, and the standard one on [-2, 2]. */
std::vector<BoundsRow> bounds_rows()
{
    return {{"ShiftedAndScaled", 1.0, 0.5, -1.0, 1.5},
            {"StandardOnMinusTwoToTwo", 0.0, 1.0, -2.0, 2.0}};
}

/** @brief Runs one row of bounds_rows(). */
class Small_TruncatedNormalBounds : public Small_DistributionsTest,
                                    public ::testing::WithParamInterface<BoundsRow>
{

};

TEST_P(Small_TruncatedNormalBounds, SamplesStayInsideMeanPlusStddevTimesTheCuts)
{
    const BoundsRow row = GetParam();

    // The bounds count standard deviations: mean + stddev * cut, exact in float for these rows.
    const std::vector<double> samples = draw(K_SEED, [&row](RandomOps& random, Tensor& t)
    {
        random.truncated_normal(t, row.mean, row.stddev, row.lower, row.upper);
    }, K_BOUNDED_DRAWS);

    expect_inside(samples, row.mean + row.stddev * row.lower, row.mean + row.stddev * row.upper);
}

INSTANTIATE_TEST_SUITE_P(
    , Small_TruncatedNormalBounds, ::testing::ValuesIn(bounds_rows()),
    [](const ::testing::TestParamInfo<BoundsRow>& info) { return std::string(info.param.name); });

TEST_F(Small_DistributionsTest, TruncatedNormalFollowsTheTruncatedLawInTheUpperTail)
{
    const std::vector<double> samples = draw(K_SEED, [](RandomOps& random, Tensor& t)
    {
        random.truncated_normal(t, 0.0, 1.0, 1.0, 3.0);
    });

    // The mean of N(0, 1) cut to [1, 3] is (pdf(1) - pdf(3)) / (cdf(3) - cdf(1)) = 1.510048.
    expect_inside(samples, 1.0, 3.0);
    expect_mean_matches(samples, truncated_normal_law(1.0, 3.0));
}

TEST_F(Small_DistributionsTest, TruncatedNormalDrawsAFarLowerBoundAsTheHalfOpenLaw)
{
    const std::vector<double> samples = draw(K_FAR_BOUND_SEED, [](RandomOps& random, Tensor& t)
    {
        random.truncated_normal(t, 0.0, 1.0, K_FAR_LOWER_BOUND, 2.0);
    });

    // N(0, 1) cut above 2 has mean -pdf(2) / cdf(2) = -0.055248 and variance 0.886452.
    expect_inside(samples, K_FAR_LOWER_BOUND, 2.0);
    expect_mean_matches(samples, truncated_normal_law(K_FAR_LOWER_BOUND, 2.0));
}

TEST_F(Small_DistributionsTest, ANormalWithZeroStddevDrawsTheMeanEverywhere)
{
    const std::vector<double> samples = draw(K_SEED, [](RandomOps& random, Tensor& t)
    {
        random.normal(t, 1.0, 0.0);
    }, K_BOUNDED_DRAWS);

    expect_inside(samples, 1.0, 1.0);
}

TEST_F(Small_DistributionsTest, ALogNormalWithZeroStddevDrawsExpOfTheMeanEverywhere)
{
    const std::vector<double> samples = draw(K_SEED, [](RandomOps& random, Tensor& t)
    {
        random.log_normal(t, 1.0, 0.0);
    }, K_BOUNDED_DRAWS);

    // The FLOAT32 store rounds the host exp by a few ulps of e.
    const double expected = std::exp(1.0);
    const double tolerance = absolute_tolerance<float>(K_EXP_OPERATIONS, expected);
    expect_inside(samples, expected - tolerance, expected + tolerance);
}
