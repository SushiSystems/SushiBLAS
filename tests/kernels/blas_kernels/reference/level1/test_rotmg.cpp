/**************************************************************************/
/* test_rotmg.cpp                                                         */
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
 * @file test_rotmg.cpp
 * @brief Tests that the reference rotmg builds a transformation that zeroes the second component.
 * @author Mustafa Garip
 */

#include <cmath>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_kernels/reference/level1/rotm.hpp>
#include <SushiBLAS/blas_kernels/reference/level1/rotmg.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Absolute tolerance of a component the transformation must zero.
        constexpr double K_TOLERANCE = 1e-12;

        TEST(Unit_ReferenceRotmg, TheTransformationZeroesTheSecondComponent)
        {
            for (const double y1 : {0.5, 3.0, -2.0})
            {
                SCOPED_TRACE(y1);
                double d1 = 2.0;
                double d2 = 1.0;
                double x1 = 1.5;
                std::vector<double> param(5, 0.0);

                Kernels::Reference::rotmg(&d1, &d2, &x1, &y1, param.data());

                std::vector<double> x{1.5};
                std::vector<double> y{y1};
                Kernels::Reference::rotm<double>(1, x.data(), 1, y.data(), 1, param.data());
                EXPECT_NEAR(y[0], 0.0, K_TOLERANCE);
                EXPECT_NEAR(x[0], x1, K_TOLERANCE * std::abs(x1));
            }
        }

        TEST(Unit_ReferenceRotmg, AZeroSecondComponentNeedsNoTransformation)
        {
            double d1 = 2.0;
            double d2 = 1.0;
            double x1 = 1.5;
            const double y1 = 0.0;
            std::vector<double> param(5, 9.0);

            Kernels::Reference::rotmg(&d1, &d2, &x1, &y1, param.data());

            EXPECT_EQ(param[0], -2.0);
            EXPECT_EQ(x1, 1.5);
        }
    } // namespace Test
} // namespace SushiBLAS
