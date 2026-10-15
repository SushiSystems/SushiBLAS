/**************************************************************************/
/* test_log.cpp                                                           */
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
 * @file test_log.cpp
 * @brief Tests ElementwiseOps::log against fixed expected values.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
using namespace SushiBLAS::Test;

class Medium_LogTest : public SushiBLASTest {};

TEST_F(Medium_LogTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, 3.0f, 4.0f});
    
    SushiBLAS::ElementwiseOps(*engine).log(t);
    engine->execute().value();
    
    verify_tensor(t, {0.00000f, 0.693147f, 1.098612f, 1.386294f});
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
    constexpr float K_LARGE_INPUT = 2.0f;

    /// @brief Holds log(2), the expectation for K_LARGE_INPUT.
    constexpr double K_LARGE_EXPECTED = 0.6931471805599453;

    /// @brief Holds log of {0.5, 1, 4, 10}.
    const std::vector<double> K_FLOAT64_EXPECTED = {-0.6931471805599453, 0.0, 1.3862943611198906, 2.302585092994046};

    /// @brief Holds the absolute tolerance for FLOAT32 results.
    constexpr double K_FLOAT32_TOLERANCE = 1e-6;

    /// @brief Holds the absolute tolerance for FLOAT64 results.
    constexpr double K_FLOAT64_TOLERANCE = 1e-12;
}

TEST_F(Medium_LogTest, ZeroInSingleElement)
{
    auto t = engine->create_tensor({1});
    fill_tensor(t, {0.0f});

    SushiBLAS::ElementwiseOps(*engine).log(t);
    engine->execute().value();

    // log(0) is -inf.
    verify_tensor(t, std::vector<double>{-K_INF}, K_FLOAT32_TOLERANCE);
}

TEST_F(Medium_LogTest, NegativeInputIsNaN)
{
    auto t = engine->create_tensor({1});
    fill_tensor(t, {-1.0f});

    SushiBLAS::ElementwiseOps(*engine).log(t);
    engine->execute().value();

    // log(-1) is outside the real domain.
    verify_tensor(t, {K_NAN});
}

TEST_F(Medium_LogTest, PositiveInfinity)
{
    auto t = engine->create_tensor({1});
    fill_tensor(t, {K_INF});

    SushiBLAS::ElementwiseOps(*engine).log(t);
    engine->execute().value();

    verify_tensor(t, {K_INF});
}

TEST_F(Medium_LogTest, ThousandElements)
{
    auto t = engine->create_tensor({static_cast<int64_t>(K_LARGE_SIZE)});
    fill_tensor(t, std::vector<float>(K_LARGE_SIZE, K_LARGE_INPUT));

    SushiBLAS::ElementwiseOps(*engine).log(t);
    engine->execute().value();

    verify_tensor(t, std::vector<double>(K_LARGE_SIZE, K_LARGE_EXPECTED), K_FLOAT32_TOLERANCE);
}

TEST_F(Medium_LogTest, Float64)
{
    auto t = engine->create_tensor({4}, SushiBLAS::Core::DataType::Float64);
    fill_tensor(t, {0.5f, 1.0f, 4.0f, 10.0f});

    SushiBLAS::ElementwiseOps(*engine).log(t);
    engine->execute().value();

    verify_tensor(t, K_FLOAT64_EXPECTED, K_FLOAT64_TOLERANCE);
}
