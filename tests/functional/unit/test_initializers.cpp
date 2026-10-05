/****************************************************************/
/* test_initializers.cpp                                        */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_initializers.cpp
 * @brief Tests the Xavier and He initializers of random().
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

// We will cluster some similar initialization tests to satisfy requirements while being lean.
class Unit_InitializersTest : public SushiBLASTest {};

TEST_F(Unit_InitializersTest, XavierUniform) 
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

TEST_F(Unit_InitializersTest, XavierNormal) 
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

TEST_F(Unit_InitializersTest, HeUniform) 
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

TEST_F(Unit_InitializersTest, HeNormal) 
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
