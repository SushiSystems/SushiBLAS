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
 * @file test_distributions.cpp
 * @brief Tests the log-normal, exponential, Poisson, Bernoulli, discrete and truncated draws.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <vector>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_DistributionsTest : public SushiBLASTest {};

TEST_F(Unit_DistributionsTest, LogNormal) 
{
    const int N = 100;
    auto t = engine->create_tensor({N});
    SushiBLAS::RandomOps(*engine).log_normal(t, 0.0, 1.0);
    engine->execute();
    
    // Log-normal distribution produces positive values.
    const float* ptr = t.data_as<float>();
    for (int i = 0; i < N; ++i) {
        EXPECT_GT(ptr[i], 0.0f);
    }
}

TEST_F(Unit_DistributionsTest, Exponential) 
{
    const int N = 100;
    auto t = engine->create_tensor({N});
    SushiBLAS::RandomOps(*engine).exponential(t, 1.0);
    engine->execute();
    
    // Exponential distribution produces non-negative values.
    const float* ptr = t.data_as<float>();
    for (int i = 0; i < N; ++i) {
        EXPECT_GE(ptr[i], 0.0f);
    }
}

TEST_F(Unit_DistributionsTest, Poisson) 
{
    const int N = 100;
    auto t = engine->create_tensor({N});
    SushiBLAS::RandomOps(*engine).poisson(t, 5.0);
    engine->execute();
    
    // Poisson distribution produces non-negative integer values.
    const float* ptr = t.data_as<float>();
    for (int i = 0; i < N; ++i) {
        EXPECT_GE(ptr[i], 0.0f);
        EXPECT_EQ(ptr[i], std::floor(ptr[i]));
    }
}

TEST_F(Unit_DistributionsTest, Bernoulli) 
{
    const int N = 100;
    auto t = engine->create_tensor({N});
    SushiBLAS::RandomOps(*engine).bernoulli(t, 0.5);
    engine->execute();
    
    // Bernoulli distribution produces only 0.0 or 1.0.
    const float* ptr = t.data_as<float>();
    for (int i = 0; i < N; ++i) {
        EXPECT_TRUE(ptr[i] == 0.0f || ptr[i] == 1.0f);
    }
}

TEST_F(Unit_DistributionsTest, DiscreteUniform) 
{
    const int N = 100;
    auto t = engine->create_tensor({N});
    SushiBLAS::RandomOps(*engine).discrete_uniform(t, 0, 10);
    engine->execute();
    
    // Discrete uniform produces integer values in [min, max].
    const float* ptr = t.data_as<float>();
    for (int i = 0; i < N; ++i) {
        EXPECT_GE(ptr[i], 0.0f);
        EXPECT_LE(ptr[i], 10.0f);
        EXPECT_EQ(ptr[i], std::floor(ptr[i]));
    }
}

TEST_F(Unit_DistributionsTest, TruncatedNormalStaysInsideItsBounds)
{
    const int N = 4096;
    auto t = engine->create_tensor({N});
    SushiBLAS::RandomOps(*engine).set_seed(20261005);
    SushiBLAS::RandomOps(*engine).truncated_normal(t, 1.0, 0.5, -1.0, 1.5);
    engine->execute();

    // The bounds count standard deviations: 1.0 + 0.5 * [-1.0, 1.5] is [0.5, 1.75], exact in float.
    const float* ptr = t.data_as<float>();
    for (int i = 0; i < N; ++i) {
        EXPECT_GE(ptr[i], 0.5f) << "Sample " << i << " is below the lower bound.";
        EXPECT_LE(ptr[i], 1.75f) << "Sample " << i << " is above the upper bound.";
    }
}

TEST_F(Unit_DistributionsTest, TruncatedNormalFollowsTheTruncatedLawInTheUpperTail)
{
    const int N = 65536;
    auto t = engine->create_tensor({N});
    SushiBLAS::RandomOps(*engine).set_seed(20261005);
    SushiBLAS::RandomOps(*engine).truncated_normal(t, 0.0, 1.0, 1.0, 3.0);
    engine->execute();

    const float* ptr = t.data_as<float>();
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        EXPECT_GE(ptr[i], 1.0f) << "Sample " << i << " is below the lower bound.";
        EXPECT_LE(ptr[i], 3.0f) << "Sample " << i << " is above the upper bound.";
        sum += static_cast<double>(ptr[i]);
    }

    // The mean of N(0, 1) cut to [1, 3] is (pdf(1) - pdf(3)) / (cdf(3) - cdf(1)) = 1.510048.
    EXPECT_NEAR(sum / N, 1.510048, 0.02);
}

TEST_F(Unit_DistributionsTest, TruncatedNormalRejectsBoundsItCannotHonour)
{
    auto t = engine->create_tensor({8});
    EXPECT_THROW(
        SushiBLAS::RandomOps(*engine).truncated_normal(t, 0.0, 1.0, 2.0, -2.0),
        std::invalid_argument);
    EXPECT_THROW(
        SushiBLAS::RandomOps(*engine).truncated_normal(t, 0.0, 1.0, 1.0, 1.0),
        std::invalid_argument);
    EXPECT_THROW(
        SushiBLAS::RandomOps(*engine).truncated_normal(t, 0.0, 1.0, 15.0, 20.0),
        std::invalid_argument);
    EXPECT_THROW(
        SushiBLAS::RandomOps(*engine).truncated_normal(t, 0.0, 1.0, -20.0, -15.0),
        std::invalid_argument);

    auto z = engine->create_tensor({8}, SushiBLAS::Core::DataType::Complex32);
    EXPECT_THROW(
        SushiBLAS::RandomOps(*engine).truncated_normal(z, 0.0, 1.0, -2.0, 2.0),
        std::invalid_argument);
}

TEST_F(Unit_DistributionsTest, TruncatedNormalDrawsAFarLowerBoundAsTheHalfOpenLaw)
{
    const int N = 65536;
    auto t = engine->create_tensor({N});
    SushiBLAS::RandomOps(*engine).set_seed(20261007);
    SushiBLAS::RandomOps(*engine).truncated_normal(t, 0.0, 1.0, -1.0e30, 2.0);
    engine->execute();

    const float* ptr = t.data_as<float>();
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        ASSERT_FALSE(std::isnan(ptr[i])) << "Sample " << i << " is NaN.";
        EXPECT_LE(ptr[i], 2.0f) << "Sample " << i << " is above the upper bound.";
        sum += static_cast<double>(ptr[i]);
    }

    // N(0, 1) cut above 2: mean -pdf(2) / cdf(2) = -0.055248, variance 0.886452.
    const double standard_error = std::sqrt(0.886452 / N);
    EXPECT_NEAR(sum / N, -0.055248, 4.0 * standard_error);
}

TEST_F(Unit_DistributionsTest, EveryDistributionRejectsAnInvalidParameterAtRecordTime)
{
    auto t = engine->create_tensor({4, 4});
    SushiBLAS::RandomOps rng(*engine);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    // No execute() follows, so a throw here comes from the recording call itself.
    EXPECT_THROW(rng.exponential(t, 0.0), std::invalid_argument);
    EXPECT_THROW(rng.exponential(t, -1.0), std::invalid_argument);
    EXPECT_THROW(rng.exponential(t, inf), std::invalid_argument);
    EXPECT_THROW(rng.exponential(t, nan), std::invalid_argument);

    EXPECT_THROW(rng.normal(t, 0.0, -1.0), std::invalid_argument);
    EXPECT_THROW(rng.normal(t, 0.0, nan), std::invalid_argument);
    EXPECT_THROW(rng.normal(t, inf, 1.0), std::invalid_argument);

    EXPECT_THROW(rng.log_normal(t, 0.0, -1.0), std::invalid_argument);
    EXPECT_THROW(rng.log_normal(t, 0.0, inf), std::invalid_argument);
    EXPECT_THROW(rng.log_normal(t, nan, 1.0), std::invalid_argument);

    EXPECT_THROW(rng.bernoulli(t, nan), std::invalid_argument);
    EXPECT_THROW(rng.bernoulli(t, -0.5), std::invalid_argument);
    EXPECT_THROW(rng.bernoulli(t, 1.5), std::invalid_argument);

    EXPECT_THROW(rng.he_uniform(t, 0), std::invalid_argument);
    EXPECT_THROW(rng.he_normal(t, -4), std::invalid_argument);
    EXPECT_THROW(rng.xavier_uniform(t, 0, 0), std::invalid_argument);
    EXPECT_THROW(rng.xavier_normal(t, 2, -2), std::invalid_argument);

    // A zero stddev is a valid degenerate law, so recording it succeeds.
    EXPECT_NO_THROW(rng.normal(t, 1.0, 0.0));
    EXPECT_NO_THROW(rng.log_normal(t, 1.0, 0.0));
}

TEST_F(Unit_DistributionsTest, EveryDistributionRejectsAnUnsupportedDtypeAtRecordTime)
{
    using SushiBLAS::Core::DataType;
    SushiBLAS::RandomOps rng(*engine);

    // No execute() follows, so a throw here comes from the recording call itself.
    for (DataType dtype : {DataType::Half, DataType::Complex32, DataType::Complex64})
    {
        auto t = engine->create_tensor({4, 4}, dtype);
        EXPECT_THROW(rng.log_normal(t), std::invalid_argument);
        EXPECT_THROW(rng.exponential(t), std::invalid_argument);
        EXPECT_THROW(rng.poisson(t), std::invalid_argument);
        EXPECT_THROW(rng.bernoulli(t), std::invalid_argument);
        EXPECT_THROW(rng.discrete_uniform(t, 0, 10), std::invalid_argument);
        EXPECT_THROW(rng.truncated_normal(t), std::invalid_argument);
        EXPECT_THROW(rng.xavier_uniform(t, 4, 4), std::invalid_argument);
        EXPECT_THROW(rng.xavier_normal(t, 4, 4), std::invalid_argument);
        EXPECT_THROW(rng.he_uniform(t, 4), std::invalid_argument);
        EXPECT_THROW(rng.he_normal(t, 4), std::invalid_argument);
        EXPECT_THROW(rng.orthogonal(t), std::invalid_argument);
    }

    // uniform and normal draw complex values, so only HALF is refused.
    auto half = engine->create_tensor({4}, DataType::Half);
    EXPECT_THROW(rng.uniform(half), std::invalid_argument);
    EXPECT_THROW(rng.normal(half), std::invalid_argument);

    // shuffle moves values, so it refuses only the integer dtypes.
    for (DataType dtype : {DataType::Int32, DataType::Int64})
    {
        auto t = engine->create_tensor({4}, dtype);
        EXPECT_THROW(rng.shuffle(t), std::invalid_argument);
    }
}
