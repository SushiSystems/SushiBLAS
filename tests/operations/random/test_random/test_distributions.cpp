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
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
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

    /// @brief Holds the FLOAT32 roundings a device exp may differ from the host one by.
    constexpr std::size_t K_EXP_OPERATIONS = 8;

    /// @brief Holds a quiet NaN.
    constexpr double K_NAN = std::numeric_limits<double>::quiet_NaN();

    /// @brief Holds positive infinity.
    constexpr double K_INF = std::numeric_limits<double>::infinity();

    /// @brief Holds a lower bound far below any draw; the kernel clamps it to -40.
    constexpr double K_FAR_LOWER_BOUND = -1.0e30;

    /** @brief Records one operation on a tensor. */
    using Record = std::function<void(RandomOps&, Tensor&)>;

    /** @brief Names one recorded call, for a row of a table of refusals. */
    struct CallRow
    {
        const char* name; /**< The row name a failure shows. */
        Record record;    /**< Records the call. */
    };

    /** @brief Returns the name of a dtype as a test name. */
    std::string name_of(DataType dtype)
    {
        return std::string(SushiBLAS::Core::to_string(dtype));
    }
}

class Medium_DistributionsTest : public SushiBLASTest
{
    protected:
        /** @brief Records one fill on count FLOAT32 elements from seed, runs it and returns the elements widened. */
        std::vector<double> draw(uint64_t seed, const Record& record, int64_t count = K_DRAWS)
        {
            auto t = engine->create_tensor({count});
            RandomOps random(recorder(), engine->rng());
            random.set_seed(seed);
            record(random, t);
            recording->run().value();
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

TEST_F(Medium_DistributionsTest, LogNormalMomentsMatchItsLaw)
{
    const std::vector<double> samples = draw(K_SEED, [](RandomOps& random, Tensor& t)
    {
        random.log_normal(t, 0.0, K_LOG_NORMAL_STDDEV);
    });

    expect_moments_match(samples, log_normal_law(0.0, K_LOG_NORMAL_STDDEV));
}

TEST_F(Medium_DistributionsTest, ExponentialMomentsMatchItsLaw)
{
    const std::vector<double> samples = draw(K_SEED, [](RandomOps& random, Tensor& t)
    {
        random.exponential(t, K_EXPONENTIAL_RATE);
    });

    expect_moments_match(samples, exponential_law(K_EXPONENTIAL_RATE));
}

TEST_F(Medium_DistributionsTest, PoissonBelowTenMatchesItsMoments)
{
    const std::vector<double> samples = draw(K_SEED, [](RandomOps& random, Tensor& t)
    {
        random.poisson(t, K_KNUTH_LAMBDA);
    });

    expect_whole(samples);
    expect_moments_match(samples, poisson_law(K_KNUTH_LAMBDA));
}

TEST_F(Medium_DistributionsTest, BernoulliMomentsMatchItsLaw)
{
    const std::vector<double> samples = draw(K_SEED, [](RandomOps& random, Tensor& t)
    {
        random.bernoulli(t, K_PROBABILITY);
    });

    expect_inside(samples, 0.0, 1.0);
    expect_moments_match(samples, bernoulli_law(K_PROBABILITY));
}

TEST_F(Medium_DistributionsTest, DiscreteUniformMomentsMatchItsLaw)
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
class Medium_TruncatedNormalBounds : public Medium_DistributionsTest,
                                     public ::testing::WithParamInterface<BoundsRow>
{

};

TEST_P(Medium_TruncatedNormalBounds, SamplesStayInsideMeanPlusStddevTimesTheCuts)
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
    , Medium_TruncatedNormalBounds, ::testing::ValuesIn(bounds_rows()),
    [](const ::testing::TestParamInfo<BoundsRow>& info) { return std::string(info.param.name); });

TEST_F(Medium_DistributionsTest, TruncatedNormalFollowsTheTruncatedLawInTheUpperTail)
{
    const std::vector<double> samples = draw(K_SEED, [](RandomOps& random, Tensor& t)
    {
        random.truncated_normal(t, 0.0, 1.0, 1.0, 3.0);
    });

    // The mean of N(0, 1) cut to [1, 3] is (pdf(1) - pdf(3)) / (cdf(3) - cdf(1)) = 1.510048.
    expect_inside(samples, 1.0, 3.0);
    expect_mean_matches(samples, truncated_normal_law(1.0, 3.0));
}

TEST_F(Medium_DistributionsTest, TruncatedNormalDrawsAFarLowerBoundAsTheHalfOpenLaw)
{
    const std::vector<double> samples = draw(K_FAR_BOUND_SEED, [](RandomOps& random, Tensor& t)
    {
        random.truncated_normal(t, 0.0, 1.0, K_FAR_LOWER_BOUND, 2.0);
    });

    // N(0, 1) cut above 2 has mean -pdf(2) / cdf(2) = -0.055248 and variance 0.886452.
    expect_inside(samples, K_FAR_LOWER_BOUND, 2.0);
    expect_mean_matches(samples, truncated_normal_law(K_FAR_LOWER_BOUND, 2.0));
}

/** @brief Names a truncated normal call its record-time check refuses. */
struct RefusedBoundsRow
{
    const char* name; /**< The row name a failure shows. */
    DataType dtype;   /**< The dtype of the tensor. */
    double lower;     /**< The lower cut. */
    double upper;     /**< The upper cut. */
};

/** @brief Returns the bounds the draw cannot honour, and a complex tensor. */
std::vector<RefusedBoundsRow> refused_bounds_rows()
{
    return {{"LowerAboveUpper", DataType::Float32, 2.0, -2.0},
            {"EmptyInterval", DataType::Float32, 1.0, 1.0},
            {"IntervalFarAboveTheMean", DataType::Float32, 15.0, 20.0},
            {"IntervalFarBelowTheMean", DataType::Float32, -20.0, -15.0},
            {"ComplexTensor", DataType::Complex32, -2.0, 2.0}};
}

/** @brief Runs one row of refused_bounds_rows(). */
class Medium_TruncatedNormalRefusal : public Medium_DistributionsTest,
                                      public ::testing::WithParamInterface<RefusedBoundsRow>
{

};

TEST_P(Medium_TruncatedNormalRefusal, IsRefusedWithInvalidArgument)
{
    const RefusedBoundsRow row = GetParam();
    auto t = engine->create_tensor({8}, row.dtype);

    EXPECT_THROW(RandomOps(recorder(), engine->rng())
                     .truncated_normal(t, 0.0, 1.0, row.lower, row.upper),
                 std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    , Medium_TruncatedNormalRefusal, ::testing::ValuesIn(refused_bounds_rows()),
    [](const ::testing::TestParamInfo<RefusedBoundsRow>& info)
    { return std::string(info.param.name); });

/** @brief Returns the calls whose parameter breaks its rule; no run follows, so a throw comes from the recording call. */
std::vector<CallRow> invalid_parameter_rows()
{
    return {
        {"ExponentialZeroRate", [](RandomOps& r, Tensor& t) { r.exponential(t, 0.0); }},
        {"ExponentialNegativeRate", [](RandomOps& r, Tensor& t) { r.exponential(t, -1.0); }},
        {"ExponentialInfiniteRate", [](RandomOps& r, Tensor& t) { r.exponential(t, K_INF); }},
        {"ExponentialNaNRate", [](RandomOps& r, Tensor& t) { r.exponential(t, K_NAN); }},
        {"NormalNegativeStddev", [](RandomOps& r, Tensor& t) { r.normal(t, 0.0, -1.0); }},
        {"NormalNaNStddev", [](RandomOps& r, Tensor& t) { r.normal(t, 0.0, K_NAN); }},
        {"NormalInfiniteMean", [](RandomOps& r, Tensor& t) { r.normal(t, K_INF, 1.0); }},
        {"LogNormalNegativeStddev", [](RandomOps& r, Tensor& t) { r.log_normal(t, 0.0, -1.0); }},
        {"LogNormalInfiniteStddev", [](RandomOps& r, Tensor& t) { r.log_normal(t, 0.0, K_INF); }},
        {"LogNormalNaNMean", [](RandomOps& r, Tensor& t) { r.log_normal(t, K_NAN, 1.0); }},
        {"BernoulliNaNProbability", [](RandomOps& r, Tensor& t) { r.bernoulli(t, K_NAN); }},
        {"BernoulliNegativeProbability", [](RandomOps& r, Tensor& t) { r.bernoulli(t, -0.5); }},
        {"BernoulliProbabilityAboveOne", [](RandomOps& r, Tensor& t) { r.bernoulli(t, 1.5); }},
        {"HeUniformZeroFanIn", [](RandomOps& r, Tensor& t) { r.he_uniform(t, 0); }},
        {"HeNormalNegativeFanIn", [](RandomOps& r, Tensor& t) { r.he_normal(t, -4); }},
        {"XavierUniformZeroFans", [](RandomOps& r, Tensor& t) { r.xavier_uniform(t, 0, 0); }},
        {"XavierNormalNegativeFanOut", [](RandomOps& r, Tensor& t) { r.xavier_normal(t, 2, -2); }}};
}

/** @brief Runs one row of invalid_parameter_rows(). */
class Medium_RandomParameterRefusal : public Medium_DistributionsTest,
                                      public ::testing::WithParamInterface<CallRow>
{

};

TEST_P(Medium_RandomParameterRefusal, IsRefusedWithInvalidArgument)
{
    auto t = engine->create_tensor({4, 4});
    RandomOps random(recorder(), engine->rng());

    EXPECT_THROW(GetParam().record(random, t), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    , Medium_RandomParameterRefusal, ::testing::ValuesIn(invalid_parameter_rows()),
    [](const ::testing::TestParamInfo<CallRow>& info) { return std::string(info.param.name); });

TEST_F(Medium_DistributionsTest, ANormalWithZeroStddevDrawsTheMeanEverywhere)
{
    const std::vector<double> samples = draw(K_SEED, [](RandomOps& random, Tensor& t)
    {
        random.normal(t, 1.0, 0.0);
    }, K_BOUNDED_DRAWS);

    expect_inside(samples, 1.0, 1.0);
}

TEST_F(Medium_DistributionsTest, ALogNormalWithZeroStddevDrawsExpOfTheMeanEverywhere)
{
    const std::vector<double> samples = draw(K_SEED, [](RandomOps& random, Tensor& t)
    {
        random.log_normal(t, 1.0, 0.0);
    }, K_BOUNDED_DRAWS);

    // The device exp may differ from the host one by a few FLOAT32 roundings of e.
    const double expected = std::exp(1.0);
    const double tolerance = absolute_tolerance<float>(K_EXP_OPERATIONS, expected);
    expect_inside(samples, expected - tolerance, expected + tolerance);
}

/** @brief Runs one dtype that every value draw refuses. */
class Medium_ValueDrawDtypeRefusal : public Medium_DistributionsTest,
                                     public ::testing::WithParamInterface<DataType>
{

};

TEST_P(Medium_ValueDrawDtypeRefusal, EveryValueDrawRefusesHalfAndComplex)
{
    auto t = engine->create_tensor({4, 4}, GetParam());
    RandomOps random(recorder(), engine->rng());

    EXPECT_THROW(random.log_normal(t), std::invalid_argument);
    EXPECT_THROW(random.exponential(t), std::invalid_argument);
    EXPECT_THROW(random.poisson(t), std::invalid_argument);
    EXPECT_THROW(random.bernoulli(t), std::invalid_argument);
    EXPECT_THROW(random.discrete_uniform(t, 0, 10), std::invalid_argument);
    EXPECT_THROW(random.truncated_normal(t), std::invalid_argument);
    EXPECT_THROW(random.xavier_uniform(t, 4, 4), std::invalid_argument);
    EXPECT_THROW(random.xavier_normal(t, 4, 4), std::invalid_argument);
    EXPECT_THROW(random.he_uniform(t, 4), std::invalid_argument);
    EXPECT_THROW(random.he_normal(t, 4), std::invalid_argument);
    EXPECT_THROW(random.orthogonal(t), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    , Medium_ValueDrawDtypeRefusal,
    ::testing::Values(DataType::Half, DataType::Complex32, DataType::Complex64),
    [](const ::testing::TestParamInfo<DataType>& info) { return name_of(info.param); });

TEST_F(Medium_DistributionsTest, UniformAndNormalRefuseHalf)
{
    auto half = engine->create_tensor({4}, DataType::Half);
    RandomOps random(recorder(), engine->rng());

    EXPECT_THROW(random.uniform(half), std::invalid_argument);
    EXPECT_THROW(random.normal(half), std::invalid_argument);
}

/** @brief Runs one integer dtype that shuffle refuses. */
class Medium_ShuffleDtypeRefusal : public Medium_DistributionsTest,
                                   public ::testing::WithParamInterface<DataType>
{

};

TEST_P(Medium_ShuffleDtypeRefusal, ShuffleRefusesIntegers)
{
    auto t = engine->create_tensor({4}, GetParam());

    EXPECT_THROW(RandomOps(recorder(), engine->rng()).shuffle(t), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    , Medium_ShuffleDtypeRefusal, ::testing::Values(DataType::Int32, DataType::Int64),
    [](const ::testing::TestParamInfo<DataType>& info) { return name_of(info.param); });
