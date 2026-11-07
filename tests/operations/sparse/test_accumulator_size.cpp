/**************************************************************************/
/* test_accumulator_size.cpp                                              */
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
 * @file test_accumulator_size.cpp
 * @brief Tests that accumulator_bytes gives the size of the accumulator of each dtype.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/sparse/accumulator_size.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using Core::DataType;

            /** @brief Holds one dtype and the bytes its accumulator is expected to take. */
            struct AccumulatorSize
            {
                const char* name;   /**< The row's suffix. */
                DataType dtype;     /**< The element type of a tensor. */
                std::size_t bytes;  /**< The expected size of its accumulator. */
            };

            /**
             * @brief Returns one row per dtype: Half and Float32 accumulate in float, Float64 in
             *        double, the complex dtypes in the complex type of that width, and an integer
             *        dtype in nothing.
             */
            std::vector<AccumulatorSize> accumulator_sizes()
            {
                return {{"Half", DataType::Half, 4},          {"Float32", DataType::Float32, 4},
                        {"Float64", DataType::Float64, 8},    {"Complex32", DataType::Complex32, 8},
                        {"Complex64", DataType::Complex64, 16}, {"Int32", DataType::Int32, 0},
                        {"Int64", DataType::Int64, 0},        {"UInt32", DataType::UInt32, 0}};
            }

            /// @brief Names a row, which gtest takes as the test's suffix.
            std::string accumulator_size_name(const ::testing::TestParamInfo<AccumulatorSize>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Reads the accumulator size of one dtype at a time. */
        class Small_AccumulatorBytes : public ::testing::TestWithParam<AccumulatorSize>
        {

        };

        TEST_P(Small_AccumulatorBytes, IsTheSizeOfTheAccumulatorOfTheDtype)
        {
            EXPECT_EQ(Internal::accumulator_bytes(GetParam().dtype), GetParam().bytes);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_AccumulatorBytes, ::testing::ValuesIn(accumulator_sizes()),
                                 accumulator_size_name);
    } // namespace Test
} // namespace SushiBLAS
