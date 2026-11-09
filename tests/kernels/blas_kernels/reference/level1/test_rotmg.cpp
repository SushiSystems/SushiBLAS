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
 * @file test_rotmg.cpp
 * @brief Tests that the reference rotmg builds a transformation that zeroes the second component.
 * @author Mustafa Garip
 */

#include <cmath>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_kernels/reference/level1/rotm.hpp>
#include <SushiBLAS/blas_kernels/reference/level1/rotmg.hpp>
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Absolute tolerance: 1024 double roundings of a value of order one.
        constexpr double K_TOLERANCE = absolute_tolerance<double>(1024);

        /// @brief Runs rotmg for one second component y1 and applies the transformation it builds.
        class Small_ReferenceRotmgSecondComponent : public ::testing::TestWithParam<double> {};

        /// @brief Tests that the transformation rotmg builds zeroes the second component and keeps x1 as the first.
        TEST_P(Small_ReferenceRotmgSecondComponent, TheTransformationZeroesTheSecondComponent)
        {
            const double y1 = GetParam();
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

        INSTANTIATE_TEST_SUITE_P(SecondComponents, Small_ReferenceRotmgSecondComponent, ::testing::Values(0.5, 3.0, -2.0));

        TEST(Small_ReferenceRotmg, AZeroSecondComponentNeedsNoTransformation)
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
