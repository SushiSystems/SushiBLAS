#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

// We will cluster some similar initialization tests to satisfy requirements while being lean.
class InitializersTest : public SushiBLASTest {};

TEST_F(InitializersTest, XavierUniform) 
{
    const int N = 10;
    auto t = engine->create_tensor({N, N});
    engine->random().xavier_uniform(t, N, N);
    engine->execute().wait();
    
    const float* ptr = t.data_as<float>();
    float limit = std::sqrt(6.0f / (N + N));
    for (int i = 0; i < N * N; ++i) {
        EXPECT_GE(ptr[i], -limit);
        EXPECT_LE(ptr[i], limit);
    }
}

TEST_F(InitializersTest, XavierNormal) 
{
    const int N = 10;
    auto t = engine->create_tensor({N, N});
    engine->random().xavier_normal(t, N, N);
    engine->execute().wait();
    // Normal distribution values are unbounded, but we check for non-NaN.
    const float* ptr = t.data_as<float>();
    for (int i = 0; i < N * N; ++i) {
        EXPECT_FALSE(std::isnan(ptr[i]));
    }
}

TEST_F(InitializersTest, HeUniform) 
{
    const int N = 10;
    auto t = engine->create_tensor({N, N});
    engine->random().he_uniform(t, N);
    engine->execute().wait();
    
    const float* ptr = t.data_as<float>();
    float limit = std::sqrt(6.0f / N);
    for (int i = 0; i < N * N; ++i) {
        EXPECT_GE(ptr[i], -limit);
        EXPECT_LE(ptr[i], limit);
    }
}

TEST_F(InitializersTest, HeNormal) 
{
    const int N = 10;
    auto t = engine->create_tensor({N, N});
    engine->random().he_normal(t, N);
    engine->execute().wait();
    const float* ptr = t.data_as<float>();
    for (int i = 0; i < N * N; ++i) {
        EXPECT_FALSE(std::isnan(ptr[i]));
    }
}
