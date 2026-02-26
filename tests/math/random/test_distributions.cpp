#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class DistributionsTest : public SushiBLASTest {};

TEST_F(DistributionsTest, LogNormal) 
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

TEST_F(DistributionsTest, Exponential) 
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

TEST_F(DistributionsTest, Poisson) 
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

TEST_F(DistributionsTest, Bernoulli) 
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

TEST_F(DistributionsTest, DiscreteUniform) 
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

TEST_F(DistributionsTest, TruncatedNormal) 
{
    const int N = 100;
    auto t = engine->create_tensor({N});
    engine->random().truncated_normal(t, 0.0, 1.0, -2.0, 2.0);
    engine->execute().wait();
}
