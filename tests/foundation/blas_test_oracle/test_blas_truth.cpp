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
        TEST(Unit_BlasTruth, AnEmptySumIsZero)
        {
            const TruthValue value = TruthSum{}.value();

            EXPECT_EQ(value.value, 0.0);
            EXPECT_EQ(value.magnitude, 0.0);
        }

        TEST(Unit_BlasTruth, KeepsATermAPlainDoubleSumRoundsAway)
        {
            TruthSum truth;
            truth.add_product(1.0e16, 1.0);
            truth.add_product(1.0, 1.0);
            truth.add_product(-1.0e16, 1.0);

            double naive = 1.0e16;
            naive += 1.0;
            naive -= 1.0e16;

            EXPECT_EQ(naive, 0.0);
            EXPECT_EQ(truth.value().value, 1.0);
            EXPECT_DOUBLE_EQ(truth.value().magnitude, 2.0e16);
        }

        TEST(Unit_BlasTruth, KeepsTheRoundingErrorOfAProduct)
        {
            // (1 + 2^-30)^2 = 1 + 2^-29 + 2^-60; the last term is below a double's step at 1.
            const double a = 1.0 + 0x1p-30;
            TruthSum truth;
            truth.add_product(a, a);
            truth.add_product(-1.0, 1.0);
            truth.add_product(-0x1p-29, 1.0);

            EXPECT_EQ(truth.value().value, 0x1p-60);
        }

        TEST(Unit_BlasTruth, UnitRoundoffIsHalfTheEpsilon)
        {
            EXPECT_EQ(unit_roundoff<double>(), 0x1p-53);
            EXPECT_EQ(unit_roundoff<float>(), 0x1p-24);
            EXPECT_EQ(unit_roundoff<sycl::half>(), 0x1p-11);
        }

        TEST(Unit_BlasTruth, TheRoundingBoundGrowsWithTheRoundings)
        {
            const double unit = unit_roundoff<float>();

            EXPECT_EQ(rounding_bound(0, unit, 100.0), 0.0);
            EXPECT_DOUBLE_EQ(rounding_bound(10, unit, 1.0), 10.0 * unit / (1.0 - 10.0 * unit));
            EXPECT_LT(rounding_bound(10, unit, 1.0), rounding_bound(11, unit, 1.0));
        }
    } // namespace Test
} // namespace SushiBLAS
