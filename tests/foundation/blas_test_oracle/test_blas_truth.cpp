/**************************************************************************/
/* test_blas_truth.cpp                                                    */
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
 * @file test_blas_truth.cpp
 * @brief Tests that TruthSum recovers the rounding a double sum loses, and the rounding bound.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <limits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blas_test_oracle/blas_truth.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Expects 1 + u to round back to 1 and 1 + 1.5 u to round up, in T: u is then the
         *        largest relative error of rounding to nearest, read from T's own arithmetic.
         */
        template<typename T>
        static void expect_unit_roundoff_is_the_rounding_threshold()
        {
            const T one = static_cast<T>(1.0f);
            const double unit = unit_roundoff<T>();

            EXPECT_EQ(static_cast<double>(static_cast<T>(one + static_cast<T>(unit))), 1.0);
            EXPECT_GT(static_cast<double>(static_cast<T>(one + static_cast<T>(1.5 * unit))), 1.0);
        }

        TEST(Small_BlasTruth, ASumOfNoTermsIsZero)
        {
            TruthSum truth;
            const TruthValue value = truth.value();

            EXPECT_EQ(value.value, 0.0);
            EXPECT_EQ(value.magnitude, 0.0);
        }

        TEST(Small_BlasTruth, KeepsATermAPlainDoubleSumRoundsAway)
        {
            TruthSum truth;
            truth.add_product(1.0e16, 1.0);
            truth.add_product(1.0, 1.0);
            truth.add_product(-1.0e16, 1.0);

            EXPECT_EQ(truth.value().value, 1.0);
            EXPECT_DOUBLE_EQ(truth.value().magnitude, 2.0e16);
        }

        TEST(Small_BlasTruth, KeepsTheRoundingErrorOfAProduct)
        {
            // (1 + 2^-30)^2 = 1 + 2^-29 + 2^-60; the last term is below a double's step at 1.
            const double a = 1.0 + 0x1p-30;
            TruthSum truth;
            truth.add_product(a, a);
            truth.add_product(-1.0, 1.0);
            truth.add_product(-0x1p-29, 1.0);

            EXPECT_EQ(truth.value().value, 0x1p-60);
        }

        TEST(Small_BlasTruth, UnitRoundoffIsTheRoundingThresholdOfDouble)
        {
            expect_unit_roundoff_is_the_rounding_threshold<double>();
        }

        TEST(Small_BlasTruth, UnitRoundoffIsTheRoundingThresholdOfFloat)
        {
            expect_unit_roundoff_is_the_rounding_threshold<float>();
        }

        TEST(Small_BlasTruth, UnitRoundoffIsTheRoundingThresholdOfHalf)
        {
            expect_unit_roundoff_is_the_rounding_threshold<sycl::half>();
        }

        TEST(Small_BlasTruth, TheRoundingBoundOfZeroRoundingsIsZero)
        {
            EXPECT_EQ(rounding_bound(0, 0.01, 100.0), 0.0);
        }

        TEST(Small_BlasTruth, TheRoundingBoundIsGammaTimesTheMagnitude)
        {
            // gamma = 10 * 0.01 / (1 - 10 * 0.01) = 1 / 9, so the bound of magnitude 3 is 1 / 3.
            EXPECT_DOUBLE_EQ(rounding_bound(10, 0.01, 3.0), 1.0 / 3.0);
        }

        TEST(Small_BlasTruth, TheRoundingBoundGrowsWithTheRoundings)
        {
            const double unit = 0.01;

            EXPECT_LT(rounding_bound(10, unit, 1.0), rounding_bound(11, unit, 1.0));
        }
    } // namespace Test
} // namespace SushiBLAS
