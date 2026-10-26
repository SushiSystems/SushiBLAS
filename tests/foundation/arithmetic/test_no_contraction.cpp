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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
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
