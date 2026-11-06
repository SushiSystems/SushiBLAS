/**************************************************************************/
/* test_exp.cpp                                                           */
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
 * @file test_exp.cpp
 * @brief Tests ElementwiseOps::exp against fixed expected values.
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
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
using namespace SushiBLAS::Test;

class Medium_ExpTest : public SushiBLASTest {};

TEST_F(Medium_ExpTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, -3.0f, 0.5f});
    
    SushiBLAS::ElementwiseOps(recorder()).exp(t);
    recording->run().value();
    
    expect_tensor(t, {2.71828f, 7.38906f, 0.04979f, 1.64872f});
}

namespace
{
    /// @brief Holds positive infinity.
    const float K_INF = std::numeric_limits<float>::infinity();

    /// @brief Holds exp of {-1, 0, 0.5, 2}.
    const std::vector<double> K_FLOAT64_EXPECTED = {0.36787944117144233, 1.0, 1.6487212707001282, 7.38905609893065};

    /// @brief Holds the absolute tolerance: 2 FLOAT32 roundings of a value of order one.
    constexpr double K_FLOAT32_TOLERANCE = absolute_tolerance<float>(2);

    /// @brief Holds the absolute tolerance: 1024 FLOAT64 roundings of a value of order one.
    constexpr double K_FLOAT64_TOLERANCE = absolute_tolerance<double>(1024);
}

TEST_F(Medium_ExpTest, ZeroInSingleElement)
{
    auto t = engine->create_tensor({1});
    fill_tensor(t, {0.0f});

    SushiBLAS::ElementwiseOps(recorder()).exp(t);
    recording->run().value();

    // exp(0) is 1.
    expect_tensor(t, std::vector<double>{1.0}, K_FLOAT32_TOLERANCE);
}

TEST_F(Medium_ExpTest, PositiveInfinity)
{
    auto t = engine->create_tensor({1});
    fill_tensor(t, {K_INF});

    SushiBLAS::ElementwiseOps(recorder()).exp(t);
    recording->run().value();

    expect_tensor(t, {K_INF});
}

TEST_F(Medium_ExpTest, Float64)
{
    auto t = engine->create_tensor({4}, SushiBLAS::Core::DataType::Float64);
    fill_tensor(t, {-1.0f, 0.0f, 0.5f, 2.0f});

    SushiBLAS::ElementwiseOps(recorder()).exp(t);
    recording->run().value();

    expect_tensor(t, K_FLOAT64_EXPECTED, K_FLOAT64_TOLERANCE);
}
