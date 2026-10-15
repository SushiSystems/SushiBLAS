/**************************************************************************/
/* test_rotm.cpp                                                          */
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
 * @file test_rotm.cpp
 * @brief Tests the reference rotm under each of the four flags of its parameter.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_kernels/reference/level1/rotm.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Returns x and y after rotm with param, as {x0, x1, y0, y1}. */
        static std::vector<double> rotated(const std::vector<double>& param)
        {
            std::vector<double> x{1.0, 2.0};
            std::vector<double> y{3.0, 4.0};
            Kernels::Reference::rotm<double>(2, x.data(), 1, y.data(), 1, param.data());
            return {x[0], x[1], y[0], y[1]};
        }

        TEST(Small_ReferenceRotm, FlagMinusOneAppliesTheFullMatrix)
        {
            // H = [h11 h12; h21 h22] = [2 3; 5 7], stored as {flag, h11, h21, h12, h22}.
            EXPECT_EQ(rotated({-1.0, 2.0, 5.0, 3.0, 7.0}),
                      (std::vector<double>{11.0, 16.0, 26.0, 38.0}));
        }

        TEST(Small_ReferenceRotm, FlagZeroKeepsUnitDiagonals)
        {
            EXPECT_EQ(rotated({0.0, 99.0, 5.0, 3.0, 99.0}),
                      (std::vector<double>{10.0, 14.0, 8.0, 14.0}));
        }

        TEST(Small_ReferenceRotm, FlagOneKeepsTheUnitAndMinusOneOffDiagonals)
        {
            EXPECT_EQ(rotated({1.0, 2.0, 99.0, 99.0, 7.0}),
                      (std::vector<double>{5.0, 8.0, 20.0, 26.0}));
        }

        TEST(Small_ReferenceRotm, FlagMinusTwoLeavesBothVectors)
        {
            EXPECT_EQ(rotated({-2.0, 9.0, 9.0, 9.0, 9.0}),
                      (std::vector<double>{1.0, 2.0, 3.0, 4.0}));
        }
    } // namespace Test
} // namespace SushiBLAS
