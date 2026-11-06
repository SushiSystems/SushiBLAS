/**************************************************************************/
/* test_serial_reductions.cpp                                             */
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
 * @file test_serial_reductions.cpp
 * @brief Tests the serial host reductions a test takes its expected values from.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/serial_reductions/serial_reductions.hpp>
#include "support/tolerance.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Absolute tolerance: 1024 double roundings of a value of order one.
        constexpr double K_TOLERANCE =
            TestSupport::K_DEFAULT_THRESHOLD * 1024.0 * TestSupport::unit_roundoff<double>();

        /// @brief Five values whose sum, extremes and positions are worked out by hand.
        const std::vector<double> K_VALUES{2.0, -3.0, 5.0, 1.0, 5.0};

        TEST(Small_SerialReductions, SumAddsTheValues)
        {
            EXPECT_EQ(ReferenceKernels::sum(K_VALUES), 10.0);
        }

        TEST(Small_SerialReductions, MeanDividesTheSumByTheCount)
        {
            EXPECT_EQ(ReferenceKernels::mean(K_VALUES), 2.0);
        }

        TEST(Small_SerialReductions, ProdMultipliesTheValues)
        {
            EXPECT_EQ(ReferenceKernels::prod(K_VALUES), -150.0);
        }

        TEST(Small_SerialReductions, MaxIsTheLargestValue)
        {
            EXPECT_EQ(ReferenceKernels::max(K_VALUES), 5.0);
        }

        TEST(Small_SerialReductions, MinIsTheSmallestValue)
        {
            EXPECT_EQ(ReferenceKernels::min(K_VALUES), -3.0);
        }

        TEST(Small_SerialReductions, ArgmaxIsTheFirstPositionOfTheLargestValue)
        {
            EXPECT_EQ(ReferenceKernels::argmax(K_VALUES), 2.0);
        }

        TEST(Small_SerialReductions, ArgminIsThePositionOfTheSmallestValue)
        {
            EXPECT_EQ(ReferenceKernels::argmin(K_VALUES), 1.0);
        }

        TEST(Small_SerialReductions, VarIsThePopulationVariance)
        {
            EXPECT_NEAR(ReferenceKernels::var(K_VALUES), 8.8, K_TOLERANCE);
        }

        TEST(Small_SerialReductions, StdDevIsTheRootOfTheVariance)
        {
            EXPECT_NEAR(ReferenceKernels::std_dev(K_VALUES), std::sqrt(8.8), K_TOLERANCE);
        }

        TEST(Small_SerialReductions, AsumAddsTheMagnitudes)
        {
            EXPECT_EQ(ReferenceKernels::asum(K_VALUES), 16.0);
        }

        TEST(Small_SerialReductions, Nrm2IsTheEuclideanLength)
        {
            EXPECT_EQ(ReferenceKernels::nrm2(std::vector<double>{3.0, 4.0}), 5.0);
        }

        TEST(Small_SerialReductions, LogSumExpShiftsByTheMaximum)
        {
            const std::vector<double> large{1000.0, 1000.0};

            EXPECT_NEAR(ReferenceKernels::logsumexp(large), 1000.0 + std::log(2.0), K_TOLERANCE);
        }

        TEST(Small_SerialReductions, AnEmptyInputGivesEachIdentity)
        {
            const std::vector<double> empty;

            EXPECT_EQ(ReferenceKernels::sum(empty), 0.0);
            EXPECT_EQ(ReferenceKernels::mean(empty), 0.0);
            EXPECT_EQ(ReferenceKernels::prod(empty), 1.0);
            EXPECT_EQ(ReferenceKernels::max(empty), std::numeric_limits<double>::lowest());
            EXPECT_EQ(ReferenceKernels::min(empty), std::numeric_limits<double>::max());
        }
    } // namespace Test
} // namespace SushiBLAS
