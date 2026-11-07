/**************************************************************************/
/* test_constant.cpp                                                      */
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
 * @file test_constant.cpp
 * @brief Tests that RandomOps::constant fills a tensor of every supported dtype with one value.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
using namespace SushiBLAS::Test;
using SushiBLAS::Core::DataType;

namespace
{
    /// @brief Holds the fill value for floating-point and complex tensors; exact in every one of them.
    constexpr double K_REAL_VALUE = 2.5;

    /// @brief Holds the fill value for integer tensors.
    constexpr double K_INTEGER_VALUE = -3.0;

    /// @brief Holds an odd element count that is not a multiple of any vector width.
    constexpr int64_t K_ODD_SIZE = 1001;
}

class Small_ConstantTest : public HostEngineTest
{
    protected:
        /**
         * @brief Fills a tensor of the given dtype and size with value and checks every element.
         * @param value Expected back exactly; complex elements expect a zero imaginary part.
         */
        void expect_constant_fill(DataType dtype, int64_t size, double value)
        {
            auto t = engine->create_tensor({size}, dtype);
            SushiBLAS::RandomOps(recorder(), engine->rng()).constant(t, value);
            ASSERT_TRUE(recording->run_succeeds());
            expect_tensor_equal(t, std::vector<double>(static_cast<size_t>(size), value));
        }
};

/** @brief Names one dtype, element count and fill value of the constant fill. */
struct ConstantRow
{
    const char* name;  /**< The row name a failure shows. */
    DataType dtype;    /**< The element type of the tensor. */
    int64_t size;      /**< The element count. */
    double value;      /**< The value every element must hold. */
};

/** @brief Returns one row per dtype at an odd size, and the one-element fill. */
std::vector<ConstantRow> constant_rows()
{
    return {{"Half", DataType::Half, K_ODD_SIZE, K_REAL_VALUE},
            {"Float32", DataType::Float32, K_ODD_SIZE, K_REAL_VALUE},
            {"Float64", DataType::Float64, K_ODD_SIZE, K_REAL_VALUE},
            {"Complex32WithZeroImaginaryPart", DataType::Complex32, K_ODD_SIZE, K_REAL_VALUE},
            {"Complex64WithZeroImaginaryPart", DataType::Complex64, K_ODD_SIZE, K_REAL_VALUE},
            {"Int32", DataType::Int32, K_ODD_SIZE, K_INTEGER_VALUE},
            {"Int64", DataType::Int64, K_ODD_SIZE, K_INTEGER_VALUE},
            {"SingleElement", DataType::Float32, 1, K_REAL_VALUE}};
}

/** @brief Runs one row of constant_rows(). */
class Small_ConstantFill : public Small_ConstantTest,
                           public ::testing::WithParamInterface<ConstantRow>
{

};

TEST_P(Small_ConstantFill, WritesTheValueToEveryElement)
{
    const ConstantRow row = GetParam();

    expect_constant_fill(row.dtype, row.size, row.value);
}

INSTANTIATE_TEST_SUITE_P(
    , Small_ConstantFill, ::testing::ValuesIn(constant_rows()),
    [](const ::testing::TestParamInfo<ConstantRow>& info) { return std::string(info.param.name); });
