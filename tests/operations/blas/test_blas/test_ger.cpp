/**************************************************************************/
/* test_ger.cpp                                                           */
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
 * @file test_ger.cpp
 * @brief Tests BLASOps::ger against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
using namespace SushiBLAS::Test;

class Medium_GERTest : public SushiBLASTest {};

TEST_F(Medium_GERTest, SimpleGER) 
{
    const int M = 2, N = 3;
    auto X = engine->create_tensor({M});
    auto Y = engine->create_tensor({N});
    auto A = engine->create_tensor({M, N});

    fill_tensor(X, {1, 2});
    fill_tensor(Y, {1, 2, 3});
    
    // Initial A is zeros
    fill_tensor(A, {0, 0, 0, 0, 0, 0});

    SushiBLAS::BLASOps(recorder()).ger(X, Y, A);
    
    recording->run().value();

    // A = x * y^T = [1; 2] * [1 2 3] = [1 2 3; 2 4 6].
    expect_tensor(A, {1, 2, 3, 2, 4, 6});
}
