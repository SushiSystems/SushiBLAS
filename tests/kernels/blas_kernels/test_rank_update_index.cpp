/**************************************************************************/
/* test_rank_update_index.cpp                                             */
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

/**
 * @file test_rank_update_index.cpp
 * @brief Tests the layout offset, work-item split and triangle tests of the rank updates against hand-derived values and an element cover.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_kernels/rank_update_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Indexing = Kernels::Indexing;

        namespace
        {
            /** @brief Holds a layout, an element, a leading dimension and the offset it has. */
            struct OffsetRow
            {
                bool row_major;
                int64_t i;
                int64_t j;
                int64_t ld;
                int64_t expected;
            };

            /** @brief Holds a work-item id pair, the axis flag and the element it names. */
            struct CoordRow
            {
                int64_t first;
                int64_t second;
                bool by_columns;
                int64_t row;
                int64_t col;
            };

            /** @brief Holds a triangle, an element and whether the element lies outside it. */
            struct OutsideRow
            {
                bool upper;
                int64_t i;
                int64_t j;
                bool outside;
            };

            /** @brief Holds a triangle and a block of rows x cols at (row0, col0). */
            struct BlockRow
            {
                bool upper;
                int64_t row0;
                int64_t rows;
                int64_t col0;
                int64_t cols;
                bool touches;
            };

            /** @brief Checks that the block touches the triangle exactly when one element is inside. */
            void expect_block_matches_its_elements(const BlockRow& row)
            {
                bool any_inside = false;
                for (int64_t i = row.row0; i < row.row0 + row.rows; ++i)
                    for (int64_t j = row.col0; j < row.col0 + row.cols; ++j)
                        any_inside = any_inside || !Indexing::outside_triangle(row.upper, i, j);
                EXPECT_EQ(Indexing::block_touches_triangle(row.upper, row.row0, row.rows, row.col0,
                                                           row.cols),
                          any_inside);
            }
        } // namespace

        class Small_RankUpdateIndexOffset : public ::testing::TestWithParam<OffsetRow>
        {
        };

        TEST_P(Small_RankUpdateIndexOffset, ARowMajorOffsetStepsByLdDownAndOneAcross)
        {
            const OffsetRow row = GetParam();
            EXPECT_EQ(Indexing::layout_offset(row.row_major, row.i, row.j, row.ld), row.expected);
        }

        // Row-major (2, 3) with ld 5 is 2*5 + 3 = 13; column-major is 2 + 3*5 = 17.
        INSTANTIATE_TEST_SUITE_P(, Small_RankUpdateIndexOffset,
                                 ::testing::Values(OffsetRow{true, 2, 3, 5, 13},
                                                   OffsetRow{false, 2, 3, 5, 17},
                                                   OffsetRow{true, 0, 0, 1, 0},
                                                   OffsetRow{false, 0, 4, 9, 36},
                                                   OffsetRow{true, 4, 0, 9, 36},
                                                   OffsetRow{false, 4, 0, 9, 4}));

        class Small_RankUpdateIndexCoord : public ::testing::TestWithParam<CoordRow>
        {
        };

        TEST_P(Small_RankUpdateIndexCoord, TheByColumnsFlagSwapsTheTwoIds)
        {
            const CoordRow row = GetParam();
            const Indexing::GridCoord got =
                Indexing::element_coord(row.first, row.second, row.by_columns);
            EXPECT_EQ(got.row, row.row);
            EXPECT_EQ(got.col, row.col);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_RankUpdateIndexCoord,
                                 ::testing::Values(CoordRow{2, 3, false, 2, 3},
                                                   CoordRow{2, 3, true, 3, 2},
                                                   CoordRow{0, 0, false, 0, 0},
                                                   CoordRow{0, 0, true, 0, 0},
                                                   CoordRow{0, 8, true, 8, 0}));

        class Small_RankUpdateIndexOutside : public ::testing::TestWithParam<OutsideRow>
        {
        };

        TEST_P(Small_RankUpdateIndexOutside, TheUpperTriangleKeepsColumnsFromTheDiagonalOn)
        {
            const OutsideRow row = GetParam();
            EXPECT_EQ(Indexing::outside_triangle(row.upper, row.i, row.j), row.outside);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_RankUpdateIndexOutside,
                                 ::testing::Values(OutsideRow{true, 2, 1, true},
                                                   OutsideRow{true, 2, 2, false},
                                                   OutsideRow{true, 2, 3, false},
                                                   OutsideRow{false, 2, 3, true},
                                                   OutsideRow{false, 2, 2, false},
                                                   OutsideRow{false, 2, 1, false},
                                                   OutsideRow{true, 0, 0, false},
                                                   OutsideRow{false, 0, 0, false}));

        class Small_RankUpdateIndexBlock : public ::testing::TestWithParam<BlockRow>
        {
        };

        TEST_P(Small_RankUpdateIndexBlock, ABlockTouchesTheTriangleAtItsNearestCorner)
        {
            const BlockRow row = GetParam();
            EXPECT_EQ(Indexing::block_touches_triangle(row.upper, row.row0, row.rows, row.col0,
                                                       row.cols),
                      row.touches);
        }

        TEST_P(Small_RankUpdateIndexBlock, ABlockTouchesTheTriangleWhenOneOfItsElementsIsInside)
        {
            expect_block_matches_its_elements(GetParam());
        }

        // Four-wide blocks: the upper triangle reaches rows up to 3, the lower one columns up to 3.
        INSTANTIATE_TEST_SUITE_P(, Small_RankUpdateIndexBlock,
                                 ::testing::Values(BlockRow{true, 0, 4, 0, 4, true},
                                                   BlockRow{true, 3, 4, 0, 4, true},
                                                   BlockRow{true, 4, 4, 0, 4, false},
                                                   BlockRow{true, 5, 4, 0, 4, false},
                                                   BlockRow{false, 0, 4, 4, 4, false},
                                                   BlockRow{false, 0, 4, 3, 4, true},
                                                   BlockRow{false, 4, 4, 0, 4, true},
                                                   BlockRow{true, 0, 1, 0, 1, true},
                                                   BlockRow{false, 0, 1, 1, 1, false},
                                                   BlockRow{true, 0, 0, 0, 0, false}));
    } // namespace Test
} // namespace SushiBLAS
