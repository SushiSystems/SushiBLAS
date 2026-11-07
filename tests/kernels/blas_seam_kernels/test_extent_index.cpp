/**************************************************************************/
/* test_extent_index.cpp                                                  */
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
 * @file test_extent_index.cpp
 * @brief Tests the block counts, tail guards and tile offsets that the kernels of this module share, against hand values and a cover check.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_seam_kernels/extent_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Indexing = Kernels::Indexing;

        namespace
        {
            /** @brief Holds an extent and the block size that covers it. */
            struct ExtentRow
            {
                int64_t extent;
                int64_t block;
            };

            /** @brief Checks that the live counts of the blocks of an extent add up to the extent. */
            void expect_live_counts_sum_to_the_extent(const ExtentRow& row)
            {
                int64_t covered = 0;
                const int64_t blocks = Indexing::ceil_div(row.extent, row.block);
                for (int64_t g = 0; g < blocks; ++g)
                    covered += Indexing::live_count(row.extent, g * row.block, row.block);
                EXPECT_EQ(covered, row.extent);
            }
        } // namespace

        class Small_ExtentIndexCover : public ::testing::TestWithParam<ExtentRow>
        {
        };

        TEST_P(Small_ExtentIndexCover, TheLiveCountsOfTheBlocksAddUpToTheExtent)
        {
            expect_live_counts_sum_to_the_extent(GetParam());
        }

        // Sizes 0, 1, 2, 3, 5 and 9 against blocks of 1, 2, 4 and 6, so every tail length occurs.
        INSTANTIATE_TEST_SUITE_P(, Small_ExtentIndexCover,
                                 ::testing::Values(ExtentRow{0, 4}, ExtentRow{1, 4}, ExtentRow{2, 4},
                                                   ExtentRow{3, 4}, ExtentRow{5, 4}, ExtentRow{9, 4},
                                                   ExtentRow{9, 1}, ExtentRow{5, 2}, ExtentRow{9, 6},
                                                   ExtentRow{3, 6}));

        /** @brief Holds one ceil_div call and the count its definition gives. */
        struct CeilDivRow
        {
            int64_t extent;
            int64_t block;
            int64_t expected;
        };

        class Small_ExtentIndexCeilDiv : public ::testing::TestWithParam<CeilDivRow>
        {
        };

        TEST_P(Small_ExtentIndexCeilDiv, CeilDivIsTheSmallestBlockCountThatCoversTheExtent)
        {
            const CeilDivRow row = GetParam();

            EXPECT_EQ(Indexing::ceil_div(row.extent, row.block), row.expected);
        }

        // ceil(extent / block): 0, 1, 1, 1, 1, 2, 3, 9, 2, 2.
        INSTANTIATE_TEST_SUITE_P(, Small_ExtentIndexCeilDiv,
                                 ::testing::Values(CeilDivRow{0, 4, 0}, CeilDivRow{1, 4, 1},
                                                   CeilDivRow{2, 4, 1}, CeilDivRow{3, 4, 1},
                                                   CeilDivRow{4, 4, 1}, CeilDivRow{5, 4, 2},
                                                   CeilDivRow{9, 4, 3}, CeilDivRow{9, 1, 9},
                                                   CeilDivRow{8, 4, 2}, CeilDivRow{3, 2, 2}));

        /** @brief Holds one live_count call and the count its definition gives. */
        struct LiveCountRow
        {
            int64_t total;
            int64_t first;
            int64_t width;
            int64_t expected;
        };

        class Small_ExtentIndexLive : public ::testing::TestWithParam<LiveCountRow>
        {
        };

        TEST_P(Small_ExtentIndexLive, LiveCountIsTheSmallerOfTheWidthAndWhatIsLeft)
        {
            const LiveCountRow row = GetParam();

            EXPECT_EQ(Indexing::live_count(row.total, row.first, row.width), row.expected);
        }

        // min(total - first, width): a whole block, the last partial block, an empty tail, a wide block.
        INSTANTIATE_TEST_SUITE_P(, Small_ExtentIndexLive,
                                 ::testing::Values(LiveCountRow{9, 0, 4, 4}, LiveCountRow{9, 4, 4, 4},
                                                   LiveCountRow{9, 8, 4, 1}, LiveCountRow{5, 4, 4, 1},
                                                   LiveCountRow{5, 0, 9, 5}, LiveCountRow{3, 3, 4, 0},
                                                   LiveCountRow{0, 0, 4, 0}, LiveCountRow{9, 8, 1, 1},
                                                   LiveCountRow{2, 0, 2, 2}, LiveCountRow{3, 0, 2, 2}));

        /** @brief Holds one row_major_offset call and the offset its definition gives. */
        struct OffsetRow
        {
            int64_t row;
            int64_t stride;
            int64_t col;
            int64_t expected;
        };

        class Small_ExtentIndexOffset : public ::testing::TestWithParam<OffsetRow>
        {
        };

        TEST_P(Small_ExtentIndexOffset, RowMajorOffsetIsRowTimesStridePlusColumn)
        {
            const OffsetRow row = GetParam();

            EXPECT_EQ(Indexing::row_major_offset(row.row, row.stride, row.col), row.expected);
        }

        // row * stride + col: 0, 5, 7, 5, 90.
        INSTANTIATE_TEST_SUITE_P(, Small_ExtentIndexOffset,
                                 ::testing::Values(OffsetRow{0, 5, 0, 0}, OffsetRow{1, 5, 0, 5},
                                                   OffsetRow{2, 3, 1, 7}, OffsetRow{3, 1, 2, 5},
                                                   OffsetRow{9, 9, 9, 90}));
    } // namespace Test
} // namespace SushiBLAS
