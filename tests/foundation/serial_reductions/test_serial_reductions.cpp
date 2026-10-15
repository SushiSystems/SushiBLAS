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
 * @file test_serial_reductions.cpp
 * @brief Tests the serial host reductions a test takes its expected values from.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/serial_reductions/serial_reductions.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Absolute tolerance of a double result against its hand value.
        constexpr double K_TOLERANCE = 1e-12;

        /// @brief Five values whose sum, extremes and positions are worked out by hand.
        const std::vector<double> K_VALUES{2.0, -3.0, 5.0, 1.0, 5.0};

        TEST(Small_SerialReductions, SumMeanAndProductOfFiveValues)
        {
            EXPECT_EQ(ReferenceKernels::sum(K_VALUES), 10.0);
            EXPECT_EQ(ReferenceKernels::mean(K_VALUES), 2.0);
            EXPECT_EQ(ReferenceKernels::prod(K_VALUES), -150.0);
        }

        TEST(Small_SerialReductions, ExtremesAndTheFirstPositionOfEach)
        {
            EXPECT_EQ(ReferenceKernels::max(K_VALUES), 5.0);
            EXPECT_EQ(ReferenceKernels::min(K_VALUES), -3.0);
            EXPECT_EQ(ReferenceKernels::argmax(K_VALUES), 2.0);
            EXPECT_EQ(ReferenceKernels::argmin(K_VALUES), 1.0);
        }

        TEST(Small_SerialReductions, SpreadAndNorms)
        {
            EXPECT_NEAR(ReferenceKernels::var(K_VALUES), 8.8, K_TOLERANCE);
            EXPECT_NEAR(ReferenceKernels::std_dev(K_VALUES), std::sqrt(8.8), K_TOLERANCE);
            EXPECT_EQ(ReferenceKernels::asum(K_VALUES), 16.0);
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
