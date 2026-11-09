/**************************************************************************/
/* test_sub.cpp                                                           */
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
 * @file test_sub.cpp
 * @brief Tests ElementwiseOps::sub against fixed expected values.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <limits>
#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
using namespace SushiBLAS::Test;

class Small_SubTest : public HostEngineTest {};

/**
 * @brief Tests that the recorded sub writes the difference of the two operands when the task runs.
 */
TEST_F(Small_SubTest, SubWritesTheDifferenceOfTheTwoOperands) 
{
    auto a = engine->create_tensor({4});
    auto b = engine->create_tensor({4});
    auto c = engine->create_tensor({4});
    fill_tensor(a, {1.0f, 2.0f, -3.0f, 0.5f});
    fill_tensor(b, {2.0f, 0.5f, 4.0f, -1.5f});
    
    SushiBLAS::ElementwiseOps(recorder()).sub(a, b, c);
    ASSERT_TRUE(recording->run_succeeds());
    
    expect_tensor(c, {-1.00000f, 1.50000f, -7.00000f, 2.00000f});
}

/// @brief Runs sub on the integer element types.
template<typename Int>
class Small_SubIntegerTest : public HostEngineTest {};

/// @brief The integer element types under test.
using SubIntegerTypes = ::testing::Types<int32_t, int64_t>;
TYPED_TEST_SUITE(Small_SubIntegerTest, SubIntegerTypes, DefaultTypeNames);

/**
 * @brief Tests that the recorded sub writes the difference of integer operands, MIN minus -1 giving MIN + 1.
 */
TYPED_TEST(Small_SubIntegerTest, SubWritesTheDifferenceOfIntegerOperands)
{
    using Int = TypeParam;
    constexpr Int k_max = std::numeric_limits<Int>::max();
    constexpr Int k_min = std::numeric_limits<Int>::min();

    auto a = this->template create_initialized_tensor<Int>({4}, {7, -3, k_max, k_min});
    auto b = this->template create_initialized_tensor<Int>({4}, {5, 4, 1, -1});
    auto c = this->engine->create_tensor({4}, sushiblas_dtype<Int>());

    SushiBLAS::ElementwiseOps(this->recorder()).sub(a, b, c);
    ASSERT_TRUE(this->recording->run_succeeds());

    EXPECT_EQ((std::vector<Int>{2, -7, k_max - 1, k_min + 1}), read_tensor<Int>(c));
}
