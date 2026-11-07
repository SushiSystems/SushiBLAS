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
 * @file test_ceilings_statistics.cpp
 * @brief Tests the median and the spread of the timed runs of a ceiling.
 * @author Mustafa Garip
 * @see tests/applications/benchmarks/device_ceilings/README.md
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <ceilings_statistics.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Names the runs of one summary and the median they give. */
        struct MedianRow
        {
            const char* name;           /**< The row name a failure shows. */
            std::vector<double> runs;   /**< The measured runs, in any order. */
            double median;              /**< The median the summary reports. */
        };

        /** @brief Returns an odd count, an even count and one run. */
        std::vector<MedianRow> median_rows()
        {
            return {{"OddCountIsTheMiddleValue", {30.0, 10.0, 20.0}, 20.0},
                    {"EvenCountIsTheMeanOfTheMiddlePair", {40.0, 10.0, 20.0, 30.0}, 25.0},
                    {"SingleRunIsThatRun", {42.0}, 42.0}};
        }

        /** @brief Runs one row of median_rows(). */
        class Small_CeilingsMedian : public ::testing::TestWithParam<MedianRow>
        {

        };

        TEST_P(Small_CeilingsMedian, IsTheMiddleOfTheSortedRuns)
        {
            const MedianRow row = GetParam();

            EXPECT_DOUBLE_EQ(Ceilings::summarize(row.runs).median, row.median);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_CeilingsMedian, ::testing::ValuesIn(median_rows()),
            [](const ::testing::TestParamInfo<MedianRow>& info) { return std::string(info.param.name); });

        /** @brief Names the runs of one summary and the spread they give. */
        struct SpreadRow
        {
            const char* name;           /**< The row name a failure shows. */
            std::vector<double> runs;   /**< The measured runs. */
            double spread;              /**< The range over the median the summary reports. */
        };

        /** @brief Returns a spread, a single run and a zero median. */
        std::vector<SpreadRow> spread_rows()
        {
            return {{"IsTheRangeOverTheMedian", {90.0, 100.0, 110.0}, 0.2},
                    {"OfASingleRunIsZero", {42.0}, 0.0},
                    {"OfAZeroMedianIsZero", {0.0, 0.0, 0.0}, 0.0}};
        }

        /** @brief Runs one row of spread_rows(). */
        class Small_CeilingsSpread : public ::testing::TestWithParam<SpreadRow>
        {

        };

        TEST_P(Small_CeilingsSpread, IsTheRangeOverTheMedianOrZero)
        {
            const SpreadRow row = GetParam();

            EXPECT_DOUBLE_EQ(Ceilings::summarize(row.runs).spread, row.spread);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_CeilingsSpread, ::testing::ValuesIn(spread_rows()),
            [](const ::testing::TestParamInfo<SpreadRow>& info) { return std::string(info.param.name); });

        TEST(Small_CeilingsStatistics, NoRunGivesAZeroResult)
        {
            const Ceilings::RunStatistics statistics = Ceilings::summarize({});

            EXPECT_DOUBLE_EQ(statistics.median, 0.0);
            EXPECT_DOUBLE_EQ(statistics.spread, 0.0);
        }
    } // namespace Test
} // namespace SushiBLAS
