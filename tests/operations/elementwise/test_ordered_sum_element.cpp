/**************************************************************************/
/* test_ordered_sum_element.cpp                                           */
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
 * @file test_ordered_sum_element.cpp
 * @brief Tests the ordered_sum element functor: the type of its weights and the bits it writes.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/elementwise/ordered_sum_element.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief A weight no float holds exactly, so the product rounds.
            constexpr double K_FIRST_WEIGHT = 0.37;

            /// @brief A second weight no float holds exactly.
            constexpr double K_SECOND_WEIGHT = -1.9;

            /// @brief The two inputs of one element and the sum the kernel must give.
            struct WeightRow
            {
                float first;
                float second;
                float expected;
            };

            /// @brief Names a row by its position in the table.
            std::string row_name(const ::testing::TestParamInfo<WeightRow>& info)
            {
                return "Row" + std::to_string(info.index);
            }
        } // namespace

        /// @brief Runs the host form of the kernel on one element of two inputs.
        class Small_OrderedSumElementWeight : public ::testing::TestWithParam<WeightRow> {};

        /**
         * @brief Tests that a weight rounded to float once on the host gives the bits of the
         *        sum, with the expected bits written as hexadecimal float literals.
         */
        TEST_P(Small_OrderedSumElementWeight, ARoundedWeightKeepsTheSumsBits)
        {
            #pragma clang fp contract(off)
            const WeightRow row = GetParam();
            const float first[] = {row.first};
            const float second[] = {row.second};
            const float* inputs[] = {first, second};
            const float weights[] = {static_cast<float>(K_FIRST_WEIGHT), static_cast<float>(K_SECOND_WEIGHT)};

            EXPECT_EQ(Dispatch::ordered_sum_at<float>(inputs, weights, 2, 0), row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_OrderedSumElementWeight,
            ::testing::Values(WeightRow{1.25f, 0.1f, 0x1.170a3ep-2f},
                              WeightRow{-3.5f, 2.0e7f, -0x1.21eacp+25f},
                              WeightRow{97.03f, -0.333f, 0x1.24453ap+5f}),
            row_name);
    } // namespace Test
} // namespace SushiBLAS
