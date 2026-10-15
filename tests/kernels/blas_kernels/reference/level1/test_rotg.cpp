/**************************************************************************/
/* test_rotg.cpp                                                          */
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
 * @file test_rotg.cpp
 * @brief Tests the reference rotg on the 3-4-5 triangle, a zero second component and a zero first.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/blas_kernels/reference/level1/rotg.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_ReferenceRotg, TheThreeFourFiveTriangleRotatesToFive)
        {
            double a = 3.0;
            double b = 4.0;
            double c = 0.0;
            double s = 0.0;

            Kernels::Reference::rotg(&a, &b, &c, &s);

            EXPECT_EQ(a, 5.0);
            EXPECT_DOUBLE_EQ(c, 0.6);
            EXPECT_DOUBLE_EQ(s, 0.8);
            EXPECT_DOUBLE_EQ(b, 1.0 / 0.6);
        }

        TEST(Small_ReferenceRotg, ALargerFirstComponentStoresTheSine)
        {
            float a = -4.0f;
            float b = 3.0f;
            float c = 0.0f;
            float s = 0.0f;

            Kernels::Reference::rotg(&a, &b, &c, &s);

            EXPECT_EQ(a, -5.0f);
            EXPECT_FLOAT_EQ(c, 0.8f);
            EXPECT_FLOAT_EQ(s, -0.6f);
            EXPECT_FLOAT_EQ(b, -0.6f);
        }

        TEST(Small_ReferenceRotg, AZeroComponentGivesTheIdentityOrTheSwap)
        {
            double a = 2.0;
            double b = 0.0;
            double c = 0.0;
            double s = 7.0;
            Kernels::Reference::rotg(&a, &b, &c, &s);
            EXPECT_EQ(a, 2.0);
            EXPECT_EQ(c, 1.0);
            EXPECT_EQ(s, 0.0);

            a = 0.0;
            b = -3.0;
            Kernels::Reference::rotg(&a, &b, &c, &s);
            EXPECT_EQ(a, -3.0);
            EXPECT_EQ(c, 0.0);
            EXPECT_EQ(s, 1.0);
            EXPECT_EQ(b, 1.0);
        }
    } // namespace Test
} // namespace SushiBLAS
