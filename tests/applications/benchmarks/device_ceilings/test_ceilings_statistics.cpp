/**************************************************************************/
/* test_ceilings_statistics.cpp                                           */
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
 * @file test_ceilings_statistics.cpp
 * @brief Tests the median and the spread of the timed runs of a ceiling.
 * @author Mustafa Garip
 * @see tests/applications/benchmarks/device_ceilings/README.md
 */

#include <gtest/gtest.h>
#include <ceilings_statistics.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_CeilingsStatistics, TheMedianOfAnOddCountIsTheMiddleValue)
        {
            const Ceilings::RunStatistics statistics = Ceilings::summarize({30.0, 10.0, 20.0});

            EXPECT_DOUBLE_EQ(statistics.median, 20.0);
        }

        TEST(Small_CeilingsStatistics, TheMedianOfAnEvenCountIsTheMeanOfTheMiddlePair)
        {
            const Ceilings::RunStatistics statistics = Ceilings::summarize({40.0, 10.0, 20.0, 30.0});

            EXPECT_DOUBLE_EQ(statistics.median, 25.0);
        }

        TEST(Small_CeilingsStatistics, TheSpreadIsTheRangeOverTheMedian)
        {
            const Ceilings::RunStatistics statistics = Ceilings::summarize({90.0, 100.0, 110.0});

            EXPECT_DOUBLE_EQ(statistics.spread, 0.2);
        }

        TEST(Small_CeilingsStatistics, ASingleRunHasNoSpread)
        {
            const Ceilings::RunStatistics statistics = Ceilings::summarize({42.0});

            EXPECT_DOUBLE_EQ(statistics.median, 42.0);
            EXPECT_DOUBLE_EQ(statistics.spread, 0.0);
        }

        TEST(Small_CeilingsStatistics, NoRunGivesAZeroResult)
        {
            const Ceilings::RunStatistics statistics = Ceilings::summarize({});

            EXPECT_DOUBLE_EQ(statistics.median, 0.0);
            EXPECT_DOUBLE_EQ(statistics.spread, 0.0);
        }

        TEST(Small_CeilingsStatistics, AZeroMedianHasNoSpread)
        {
            const Ceilings::RunStatistics statistics = Ceilings::summarize({0.0, 0.0, 0.0});

            EXPECT_DOUBLE_EQ(statistics.spread, 0.0);
        }
    } // namespace Test
} // namespace SushiBLAS
