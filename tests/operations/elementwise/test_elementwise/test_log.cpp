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
 * @file test_log.cpp
 * @brief Tests ElementwiseOps::log against fixed expected values.
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

class Small_LogTest : public HostEngineTest {};

/**
 * @brief Tests that the recorded log replaces each element with its natural logarithm when the task runs.
 */
TEST_F(Small_LogTest, LogReplacesEachElementWithItsNaturalLogarithm) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, 3.0f, 4.0f});
    
    SushiBLAS::ElementwiseOps(recorder()).log(t);
    ASSERT_TRUE(recording->run_succeeds());
    
    expect_tensor(t, {0.00000f, 0.693147f, 1.098612f, 1.386294f});
}

namespace
{
    /// @brief Holds log of {0.5, 1, 4, 10}.
    const std::vector<double> K_FLOAT64_EXPECTED = {-0.6931471805599453, 0.0, 1.3862943611198906, 2.302585092994046};

    /// @brief Holds the absolute tolerance: 1024 FLOAT64 roundings of a value of order one.
    constexpr double K_FLOAT64_TOLERANCE = absolute_tolerance<double>(1024);
}

/**
 * @brief Tests that the recorded log computes in double on a FLOAT64 tensor.
 */
TEST_F(Small_LogTest, LogOnAFloat64TensorKeepsTheDoublePrecision)
{
    auto t = engine->create_tensor({4}, SushiBLAS::Core::DataType::Float64);
    fill_tensor(t, {0.5f, 1.0f, 4.0f, 10.0f});

    SushiBLAS::ElementwiseOps(recorder()).log(t);
    ASSERT_TRUE(recording->run_succeeds());

    expect_tensor(t, K_FLOAT64_EXPECTED, K_FLOAT64_TOLERANCE);
}
