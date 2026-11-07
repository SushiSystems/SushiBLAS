/**************************************************************************/
/* test_triangular_solve_index.cpp                                        */
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
 * @file test_triangular_solve_index.cpp
 * @brief Tests the block walk, trailing range and stream split of the blocked triangular solves against hand-derived values and a cover check.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_kernels/triangular_solve_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Indexing = Kernels::Indexing;

        namespace
        {
            /** @brief Holds a transpose, a stored triangle and whether op(A) is upper. */
            struct UpperRow
            {
                bool transposed;
                bool stored_upper;
                bool op_upper;
            };

            /** @brief Holds one pass of a blocked solve and the steps it takes. */
            struct BlockRow
            {
                int64_t pass;
                int64_t blocks;
                int64_t width;
                int64_t order;
                bool forward;
                int64_t first;
                int64_t last;
            };

            /** @brief Holds a blocked solve of `order` steps in `blocks` blocks of `width`. */
            struct CoverRow
            {
                int64_t order;
                int64_t width;
                int64_t blocks;
                bool forward;
            };

            /** @brief Holds a solved block, the system around it and the steps left to update. */
            struct TrailingRow
            {
                int64_t block_first;
                int64_t block_last;
                int64_t order;
                int64_t reach;
                bool forward;
                int64_t first;
                int64_t count;
            };

            /** @brief Holds a row of a block, a bound and the column they give. */
            struct ColumnRow
            {
                int64_t block;
                int64_t row;
                int64_t expected;
            };

            /** @brief Holds a stream index and the batch entry and vector it names. */
            struct StreamRow
            {
                int64_t stream;
                int64_t vectors;
                int64_t batch;
                int64_t vector;
            };

            /** @brief Holds an element of the trailing update and the unknown it names. */
            struct UnknownRow
            {
                int64_t element;
                int64_t trailing_first;
                int64_t trailing_count;
                int64_t vectors;
                bool step_is_innermost;
                int64_t batch;
                int64_t step;
                int64_t vector;
            };

            /** @brief Checks that the blocks of every pass add up to exactly the steps of the solve. */
            void expect_blocks_cover_the_order(const CoverRow& row)
            {
                int64_t covered = 0;
                for (int64_t pass = 0; pass < row.blocks; ++pass)
                {
                    const Indexing::StepRange block = Indexing::solve_block(
                        pass, row.blocks, row.width, row.order, row.forward);
                    covered += block.last - block.first;
                }
                EXPECT_EQ(covered, row.order);
            }
        } // namespace

        class Small_TriangularSolveIndexUpper : public ::testing::TestWithParam<UpperRow>
        {
        };

        TEST_P(Small_TriangularSolveIndexUpper, TransposingFlipsTheStoredTriangle)
        {
            EXPECT_EQ(Indexing::effective_upper(GetParam().transposed, GetParam().stored_upper),
                      GetParam().op_upper);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_TriangularSolveIndexUpper,
                                 ::testing::Values(UpperRow{false, true, true},
                                                   UpperRow{false, false, false},
                                                   UpperRow{true, true, false},
                                                   UpperRow{true, false, true}));

        class Small_TriangularSolveIndexBlock : public ::testing::TestWithParam<BlockRow>
        {
        };

        TEST_P(Small_TriangularSolveIndexBlock, ThePassNamesItsBlockTopDownOrBottomUp)
        {
            const BlockRow row = GetParam();
            const Indexing::StepRange got =
                Indexing::solve_block(row.pass, row.blocks, row.width, row.order, row.forward);
            EXPECT_EQ(got.first, row.first);
            EXPECT_EQ(got.last, row.last);
        }

        // Nine steps in fours: [0, 4), [4, 8), tail [8, 9), backward in reverse; three in twos: [2, 3).
        INSTANTIATE_TEST_SUITE_P(, Small_TriangularSolveIndexBlock,
                                 ::testing::Values(BlockRow{0, 3, 4, 9, true, 0, 4},
                                                   BlockRow{1, 3, 4, 9, true, 4, 8},
                                                   BlockRow{2, 3, 4, 9, true, 8, 9},
                                                   BlockRow{0, 3, 4, 9, false, 8, 9},
                                                   BlockRow{1, 3, 4, 9, false, 4, 8},
                                                   BlockRow{2, 3, 4, 9, false, 0, 4},
                                                   BlockRow{1, 2, 2, 3, true, 2, 3},
                                                   BlockRow{0, 1, 5, 5, true, 0, 5},
                                                   BlockRow{0, 1, 1, 1, false, 0, 1},
                                                   BlockRow{0, 1, 1, 0, true, 0, 0}));

        class Small_TriangularSolveIndexCover : public ::testing::TestWithParam<CoverRow>
        {
        };

        TEST_P(Small_TriangularSolveIndexCover, TheBlocksOfAllPassesAddUpToTheOrder)
        {
            expect_blocks_cover_the_order(GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_TriangularSolveIndexCover,
                                 ::testing::Values(CoverRow{0, 1, 1, true}, CoverRow{1, 1, 1, true},
                                                   CoverRow{2, 1, 2, false}, CoverRow{3, 2, 2, true},
                                                   CoverRow{5, 2, 3, false}, CoverRow{9, 4, 3, true},
                                                   CoverRow{9, 4, 3, false}, CoverRow{9, 1, 9, true},
                                                   CoverRow{5, 5, 1, false}));

        class Small_TriangularSolveIndexTrailing : public ::testing::TestWithParam<TrailingRow>
        {
        };

        TEST_P(Small_TriangularSolveIndexTrailing, TheUpdateReachesTheNearestStepsPastTheBlock)
        {
            const TrailingRow row = GetParam();
            const Indexing::TrailingRange got = Indexing::trailing_range(
                Indexing::StepRange{row.block_first, row.block_last}, row.order, row.reach,
                row.forward);
            EXPECT_EQ(got.first, row.first);
            EXPECT_EQ(got.count, row.count);
        }

        // Order 9, block [4, 8): forward leaves step 8, backward steps 0..3, a reach of 2 keeps two.
        INSTANTIATE_TEST_SUITE_P(, Small_TriangularSolveIndexTrailing,
                                 ::testing::Values(TrailingRow{4, 8, 9, 9, true, 8, 1},
                                                   TrailingRow{4, 8, 9, 9, false, 0, 4},
                                                   TrailingRow{4, 8, 9, 2, false, 2, 2},
                                                   TrailingRow{0, 4, 9, 3, true, 4, 3},
                                                   TrailingRow{0, 4, 9, 9, true, 4, 5},
                                                   TrailingRow{8, 9, 9, 9, true, 9, 0},
                                                   TrailingRow{0, 4, 9, 9, false, 0, 0},
                                                   TrailingRow{4, 8, 9, 0, true, 8, 0}));

        class Small_TriangularSolveIndexFirstColumn : public ::testing::TestWithParam<ColumnRow>
        {
        };

        TEST_P(Small_TriangularSolveIndexFirstColumn, TheLaterOfTheBlockStartAndTheRowStartWins)
        {
            EXPECT_EQ(Indexing::block_first_column(GetParam().block, GetParam().row),
                      GetParam().expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_TriangularSolveIndexFirstColumn,
                                 ::testing::Values(ColumnRow{4, 2, 4}, ColumnRow{4, 6, 6},
                                                   ColumnRow{4, 4, 4}, ColumnRow{0, 0, 0}));

        class Small_TriangularSolveIndexEndColumn : public ::testing::TestWithParam<ColumnRow>
        {
        };

        TEST_P(Small_TriangularSolveIndexEndColumn, TheEarlierOfTheBlockEndAndTheRowEndWins)
        {
            EXPECT_EQ(Indexing::block_end_column(GetParam().block, GetParam().row),
                      GetParam().expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_TriangularSolveIndexEndColumn,
                                 ::testing::Values(ColumnRow{8, 5, 5}, ColumnRow{8, 9, 8},
                                                   ColumnRow{8, 8, 8}, ColumnRow{0, 0, 0}));

        class Small_TriangularSolveIndexStream : public ::testing::TestWithParam<StreamRow>
        {
        };

        TEST_P(Small_TriangularSolveIndexStream, AStreamIsABatchEntryTimesVectorsPlusAVector)
        {
            const StreamRow row = GetParam();
            const Indexing::StreamId got = Indexing::split_stream(row.stream, row.vectors);
            EXPECT_EQ(got.batch, row.batch);
            EXPECT_EQ(got.vector, row.vector);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_TriangularSolveIndexStream,
                                 ::testing::Values(StreamRow{0, 1, 0, 0}, StreamRow{4, 1, 4, 0},
                                                   StreamRow{5, 3, 1, 2}, StreamRow{3, 3, 1, 0},
                                                   StreamRow{0, 5, 0, 0}, StreamRow{8, 5, 1, 3}));

        class Small_TriangularSolveIndexUnknown : public ::testing::TestWithParam<UnknownRow>
        {
        };

        TEST_P(Small_TriangularSolveIndexUnknown, TheInnermostIndexRunsFastestThroughTheUpdate)
        {
            const UnknownRow row = GetParam();
            const Indexing::UpdateUnknown got = Indexing::update_unknown(
                row.element, Indexing::TrailingRange{row.trailing_first, row.trailing_count},
                row.vectors, row.step_is_innermost);
            EXPECT_EQ(got.batch, row.batch);
            EXPECT_EQ(got.step, row.step);
            EXPECT_EQ(got.vector, row.vector);
        }

        // Three steps from 4, two vectors: 6 per matrix; element 4 is (step 5, vector 1) steps-first.
        INSTANTIATE_TEST_SUITE_P(, Small_TriangularSolveIndexUnknown,
                                 ::testing::Values(UnknownRow{0, 4, 3, 2, true, 0, 4, 0},
                                                   UnknownRow{4, 4, 3, 2, true, 0, 5, 1},
                                                   UnknownRow{6, 4, 3, 2, true, 1, 4, 0},
                                                   UnknownRow{11, 4, 3, 2, true, 1, 6, 1},
                                                   UnknownRow{4, 4, 3, 2, false, 0, 6, 0},
                                                   UnknownRow{5, 4, 3, 2, false, 0, 6, 1},
                                                   UnknownRow{7, 4, 3, 2, false, 1, 4, 1},
                                                   UnknownRow{0, 7, 1, 1, true, 0, 7, 0},
                                                   UnknownRow{0, 7, 1, 1, false, 0, 7, 0}));
    } // namespace Test
} // namespace SushiBLAS
