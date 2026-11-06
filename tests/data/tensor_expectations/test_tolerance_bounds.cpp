/**************************************************************************/
/* test_tolerance_bounds.cpp                                              */
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
 * @file test_tolerance_bounds.cpp
 * @brief Tests that the absolute tolerances grow with the operation count, the magnitude and the precision.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_ToleranceBounds, OneFloatRoundingOfOneIsSixteenUnitRoundoffs)
        {
            EXPECT_EQ(absolute_tolerance<float>(1), 16.0 * 0x1p-24);
        }

        TEST(Small_ToleranceBounds, TheBoundScalesWithOperationsAndMagnitude)
        {
            EXPECT_EQ(absolute_tolerance<double>(4, 8.0), 32.0 * absolute_tolerance<double>(1));
        }

        TEST(Small_ToleranceBounds, AHalfRoundingIsTwoToTheMinusElevenTimesTheThreshold)
        {
            EXPECT_EQ(half_absolute_tolerance(1), 16.0 * 0x1p-11);
        }

        TEST(Small_ToleranceBounds, TheDefaultIsOneHundredTwentyEightFloatRoundings)
        {
            EXPECT_EQ(K_DEFAULT_TOLERANCE, absolute_tolerance<float>(128));
        }
    } // namespace Test
} // namespace SushiBLAS
