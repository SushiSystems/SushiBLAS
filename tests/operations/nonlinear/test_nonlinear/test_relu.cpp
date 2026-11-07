/**************************************************************************/
/* test_relu.cpp                                                          */
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
 * @file test_relu.cpp
 * @brief Tests NonLinearOps::relu and relu_backward against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
using namespace SushiBLAS::Test;

class Medium_ReLUTest : public SushiBLASTest {};

TEST_F(Medium_ReLUTest, ForwardZeroesNegativeInputsAndKeepsTheRest) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {-2.0f, -0.5f, 0.0f, 3.5f});
    
    SushiBLAS::NonLinearOps(recorder()).relu(t);
    recording->run().value();
    
    expect_tensor(t, {0.0f, 0.0f, 0.0f, 3.5f});
}

TEST_F(Medium_ReLUTest, BackwardPassesTheGradientOnlyWhereTheInputIsPositive) 
{
    auto dy = engine->create_tensor({4});
    auto x = engine->create_tensor({4});
    auto dx = engine->create_tensor({4});
    
    fill_tensor(dy, {1.0f, 2.0f, 3.0f, 4.0f});
    fill_tensor(x, {-1.0f, 0.0f, 1.0f, 2.0f});
    fill_tensor(dx, {0.0f, 0.0f, 0.0f, 0.0f});
    
    SushiBLAS::NonLinearOps(recorder()).relu_backward(dy, x, dx);
    recording->run().value();
    
    expect_tensor(dx, {0.0f, 0.0f, 3.0f, 4.0f});
}
