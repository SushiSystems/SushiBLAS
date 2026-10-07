/**************************************************************************/
/* test_sin.cpp                                                           */
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
 * @file test_sin.cpp
 * @brief Tests elementwise().sin against fixed expected values.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_SinTest : public SushiBLASTest {};

TEST_F(Unit_SinTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, -3.0f, 0.5f});
    
    engine->elementwise().sin(t);
    engine->execute().wait();
    
    verify_tensor(t, {0.84147f, 0.90930f, -0.14112f, 0.47943f});
}

namespace
{
    /// @brief Holds positive infinity.
    const float K_INF = std::numeric_limits<float>::infinity();

    /// @brief Holds a quiet NaN, the expectation for an input outside the domain.
    const float K_NAN = std::numeric_limits<float>::quiet_NaN();

    /// @brief Holds the element count of the large case.
    constexpr size_t K_LARGE_SIZE = 1000;

    /// @brief Holds the input every element of the large case carries.
    constexpr float K_LARGE_INPUT = 1.0f;

    /// @brief Holds sin(1), the expectation for K_LARGE_INPUT.
    constexpr double K_LARGE_EXPECTED = 0.8414709848078965;

    /// @brief Holds sin of {-1, 0, 0.5, 2}.
    const std::vector<double> K_FLOAT64_EXPECTED = {-0.8414709848078965, 0.0, 0.479425538604203, 0.9092974268256817};

    /// @brief Holds the absolute tolerance for FLOAT32 results.
    constexpr double K_FLOAT32_TOLERANCE = 1e-6;

    /// @brief Holds the absolute tolerance for FLOAT64 results.
    constexpr double K_FLOAT64_TOLERANCE = 1e-12;
}

TEST_F(Unit_SinTest, ZeroInSingleElement)
{
    auto t = engine->create_tensor({1});
    fill_tensor(t, {0.0f});

    engine->elementwise().sin(t);
    engine->execute().wait();

    // sin(0) is 0.
    verify_tensor(t, std::vector<double>{0.0}, K_FLOAT32_TOLERANCE);
}

TEST_F(Unit_SinTest, PositiveInfinity)
{
    auto t = engine->create_tensor({1});
    fill_tensor(t, {K_INF});

    engine->elementwise().sin(t);
    engine->execute().wait();

    verify_tensor(t, {K_NAN});
}

TEST_F(Unit_SinTest, ThousandElements)
{
    auto t = engine->create_tensor({static_cast<int64_t>(K_LARGE_SIZE)});
    fill_tensor(t, std::vector<float>(K_LARGE_SIZE, K_LARGE_INPUT));

    engine->elementwise().sin(t);
    engine->execute().wait();

    verify_tensor(t, std::vector<double>(K_LARGE_SIZE, K_LARGE_EXPECTED), K_FLOAT32_TOLERANCE);
}

TEST_F(Unit_SinTest, Float64)
{
    auto t = engine->create_tensor({4}, SushiBLAS::Core::DataType::FLOAT64);
    fill_tensor(t, {-1.0f, 0.0f, 0.5f, 2.0f});

    engine->elementwise().sin(t);
    engine->execute().wait();

    verify_tensor(t, K_FLOAT64_EXPECTED, K_FLOAT64_TOLERANCE);
}
