/**************************************************************************/
/* test_composite_cost.cpp                                                */
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
 * @file test_composite_cost.cpp
 * @brief Tests the cost of operations composed from several tasks.
 * @author Mustafa Garip
 * @see tests/foundation/cost/README.md
 */

#include <SushiBLAS/cost/composite_cost.hpp>

#include <gtest/gtest.h>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_CompositeCost, SoftmaxCostsMaxShiftExpSumAndDivide)
        {
            // Per row of 5: max 4, shift 5, exp 5, sum 4, divide 5.
            EXPECT_EQ(softmax_flops(2, 5), 2U * (4U + 5U + 5U + 4U + 5U));
        }

        TEST(Small_CompositeCost, ARowOfOneStillShiftsExponentiatesAndDivides)
        {
            EXPECT_EQ(softmax_flops(1, 1), 3U);
        }

        TEST(Small_CompositeCost, AnEmptySoftmaxCostsNothing)
        {
            EXPECT_EQ(softmax_flops(0, 5), 0U);
            EXPECT_EQ(softmax_flops(5, 0), 0U);
        }
    } // namespace Test
} // namespace SushiBLAS
