/**************************************************************************/
/* test_sqrt.cpp                                                          */
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
 * @file test_sqrt.cpp
 * @brief Tests ElementwiseOps::sqrt against fixed expected values.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
using namespace SushiBLAS::Test;

class Small_SqrtTest : public HostEngineTest {};

/**
 * @brief Tests that the recorded sqrt replaces each element with its square root when the task runs.
 */
TEST_F(Small_SqrtTest, SqrtReplacesEachElementWithItsSquareRoot) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, 3.0f, 4.0f});
    
    SushiBLAS::ElementwiseOps(recorder()).sqrt(t);
    ASSERT_TRUE(recording->run_succeeds());
    
    expect_tensor(t, {1.00000f, 1.414213f, 1.732050f, 2.00000f});
}

namespace
{
    /// @brief Holds sqrt of {0.25, 1, 2, 9}.
    const std::vector<double> K_FLOAT64_EXPECTED = {0.5, 1.0, 1.4142135623730951, 3.0};

    /// @brief Holds the absolute tolerance: 1024 FLOAT64 roundings of a value of order one.
    constexpr double K_FLOAT64_TOLERANCE = absolute_tolerance<double>(1024);
}

/**
 * @brief Tests that the recorded sqrt computes in double on a FLOAT64 tensor.
 */
TEST_F(Small_SqrtTest, SqrtOnAFloat64TensorKeepsTheDoublePrecision)
{
    auto t = engine->create_tensor({4}, SushiBLAS::Core::DataType::Float64);
    fill_tensor(t, {0.25f, 1.0f, 2.0f, 9.0f});

    SushiBLAS::ElementwiseOps(recorder()).sqrt(t);
    ASSERT_TRUE(recording->run_succeeds());

    expect_tensor(t, K_FLOAT64_EXPECTED, K_FLOAT64_TOLERANCE);
}
