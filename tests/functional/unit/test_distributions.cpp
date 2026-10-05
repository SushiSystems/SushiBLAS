/****************************************************************/
/* test_distributions.cpp                                       */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_distributions.cpp
 * @brief Tests the log-normal, exponential, Poisson, Bernoulli, discrete and truncated draws.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_DistributionsTest : public SushiBLASTest {};

TEST_F(Unit_DistributionsTest, LogNormal) 
{
    const int N = 100;
    auto t = engine->create_tensor({N});
    engine->random().log_normal(t, 0.0, 1.0);
    engine->execute().wait();
    
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
    engine->random().exponential(t, 1.0);
    engine->execute().wait();
    
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
    engine->random().poisson(t, 5.0);
    engine->execute().wait();
    
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
    engine->random().bernoulli(t, 0.5);
    engine->execute().wait();
    
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
    engine->random().discrete_uniform(t, 0, 10);
    engine->execute().wait();
    
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
    engine->random().set_seed(20261005);
    engine->random().truncated_normal(t, 1.0, 0.5, -1.0, 1.5);
    engine->execute().wait();

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
    engine->random().set_seed(20261005);
    engine->random().truncated_normal(t, 0.0, 1.0, 1.0, 3.0);
    engine->execute().wait();

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
    EXPECT_THROW(engine->random().truncated_normal(t, 0.0, 1.0, 2.0, -2.0), std::runtime_error);
    EXPECT_THROW(engine->random().truncated_normal(t, 0.0, 1.0, 1.0, 1.0), std::runtime_error);
    EXPECT_THROW(engine->random().truncated_normal(t, 0.0, 1.0, 15.0, 20.0), std::runtime_error);
    EXPECT_THROW(engine->random().truncated_normal(t, 0.0, 1.0, -20.0, -15.0), std::runtime_error);

    auto z = engine->create_tensor({8}, SushiBLAS::Core::DataType::COMPLEX32);
    EXPECT_THROW(engine->random().truncated_normal(z, 0.0, 1.0, -2.0, 2.0), std::runtime_error);
}
