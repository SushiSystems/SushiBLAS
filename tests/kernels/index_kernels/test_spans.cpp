/**************************************************************************/
/* test_spans.cpp                                                         */
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
 * @file test_spans.cpp
 * @brief Tests MatrixSpan offsets and is_in_range against hand values.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/index_kernels/spans.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_IndexSpans, TheOffsetOfAnElementIsItsRowAndColumnTimesTheirStrides)
        {
            const Kernels::MatrixSpan span{3, 4, 10, 2};

            EXPECT_EQ(span.offset(0, 0), 0);
            EXPECT_EQ(span.offset(2, 3), 26);
        }

        /** @brief Holds one index and whether it selects a position of an extent of 4. */
        struct RangeRow
        {
            int64_t index;
            bool is_inside;
        };

        class Small_IndexSpansRange : public ::testing::TestWithParam<RangeRow>
        {
        };

        TEST_P(Small_IndexSpansRange, AnIndexSelectsAPositionOnlyInsideTheExtent)
        {
            const RangeRow row = GetParam();

            EXPECT_EQ(Kernels::is_in_range(row.index, 4), row.is_inside);
            EXPECT_EQ(Kernels::is_in_range(static_cast<int32_t>(row.index), 4), row.is_inside);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_IndexSpansRange,
                                 ::testing::Values(RangeRow{-1, false}, RangeRow{0, true},
                                                   RangeRow{3, true}, RangeRow{4, false},
                                                   RangeRow{100, false}));
    } // namespace Test
} // namespace SushiBLAS
