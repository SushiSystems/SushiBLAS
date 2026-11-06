/**************************************************************************/
/* test_no_contraction.cpp                                                */
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
 * @file test_no_contraction.cpp
 * @brief Tests that SB_FP_CONTRACT_OFF keeps a product rounded before it is added.
 * @author Mustafa Garip
 */

#include <SushiBLAS/arithmetic/no_contraction.hpp>

#include <cmath>
#include <gtest/gtest.h>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns a * b + c with contraction forbidden. */
            float unfused(float a, float b, float c)
            {
                SB_FP_CONTRACT_OFF
                return a * b + c;
            }
        } // namespace

        TEST(Small_NoContraction, RoundsTheProductBeforeTheSum)
        {
            // (1 + 2^-12)^2 rounds to 1 + 2^-11; only a fused step keeps the 2^-24 left over.
            const float a = 1.0f + std::ldexp(1.0f, -12);
            const float c = -(1.0f + std::ldexp(1.0f, -11));

            EXPECT_EQ(unfused(a, a, c), 0.0f);
            EXPECT_EQ(std::fma(a, a, c), std::ldexp(1.0f, -24));
        }
    } // namespace Test
} // namespace SushiBLAS
