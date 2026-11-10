/**************************************************************************/
/* test_gemm_index.cpp                                                    */
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
 * @file test_gemm_index.cpp
 * @brief Tests the group, staging, fold and fragment index math of the SYCL GEMM kernels against hand values and cover checks.
 * @author Mustafa Garip
 */

#include <set>
#include <tuple>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_seam_kernels/gemm_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Indexing = Kernels::Indexing;

        namespace
        {
            /** @brief Checks the row and the column of a coordinate. */
            void expect_coord(const Indexing::GridCoord& at, int64_t row, int64_t col)
            {
                EXPECT_EQ(at.row, row);
                EXPECT_EQ(at.col, col);
            }

            /** @brief Holds the extents of a staged block and how its elements are adjacent. */
            struct StagedBlock
            {
                int64_t rows;
                int64_t cols;
                int64_t lanes;
                bool columns_contiguous;
            };

            /** @brief Checks that the staging indices of a block name each of its elements once. */
            void expect_stage_coords_cover_the_block(const StagedBlock& block)
            {
                std::set<std::tuple<int64_t, int64_t>> seen;
                for (int64_t idx = 0; idx < block.rows * block.cols; ++idx)
                {
                    const Indexing::GridCoord at = Indexing::stage_coord(
                        idx, block.rows, block.cols, block.columns_contiguous);
                    EXPECT_LT(at.row, block.rows);
                    EXPECT_LT(at.col, block.cols);
                    seen.emplace(at.row, at.col);
                }
                EXPECT_EQ(static_cast<int64_t>(seen.size()), block.rows * block.cols);
            }

            /** @brief Checks that the vector loads of a block take each of its elements once. */
            void expect_vector_coords_cover_the_block(const StagedBlock& block)
            {
                std::set<std::tuple<int64_t, int64_t>> seen;
                for (int64_t v = 0; v < block.rows * block.cols / block.lanes; ++v)
                {
                    const Indexing::GridCoord at = Indexing::stage_vector_coord(
                        v, block.rows, block.cols, block.lanes, block.columns_contiguous);
                    for (int64_t l = 0; l < block.lanes; ++l)
                        seen.emplace(block.columns_contiguous ? at.row + l : at.row,
                                     block.columns_contiguous ? at.col : at.col + l);
                }
                EXPECT_EQ(static_cast<int64_t>(seen.size()), block.rows * block.cols);
                for (const std::tuple<int64_t, int64_t>& element : seen)
                {
                    EXPECT_LT(std::get<0>(element), block.rows);
                    EXPECT_LT(std::get<1>(element), block.cols);
                }
            }
        } // namespace

        class Small_GemmIndexStagedBlock : public ::testing::TestWithParam<StagedBlock>
        {
        };

        TEST_P(Small_GemmIndexStagedBlock, EveryStagingIndexNamesADistinctElementOfTheBlock)
        {
            expect_stage_coords_cover_the_block(GetParam());
        }

        TEST_P(Small_GemmIndexStagedBlock, EveryVectorLoadTakesDistinctElementsOfTheBlock)
        {
            expect_vector_coords_cover_the_block(GetParam());
        }

        // Blocks whose sides are multiples of the lane count, in both adjacency orders.
        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexStagedBlock,
                                 ::testing::Values(StagedBlock{2, 2, 2, true}, StagedBlock{2, 2, 2, false},
                                                   StagedBlock{4, 4, 4, true}, StagedBlock{4, 4, 4, false},
                                                   StagedBlock{8, 4, 4, true}, StagedBlock{8, 4, 4, false},
                                                   StagedBlock{4, 8, 4, true}, StagedBlock{4, 8, 4, false},
                                                   StagedBlock{16, 8, 8, true},
                                                   StagedBlock{16, 8, 8, false}));

        /** @brief Holds one stage_coord or stage_vector_coord call and the coordinate it must give. */
        struct StageRow
        {
            int64_t index;
            StagedBlock block;
            int64_t row;
            int64_t col;
        };

        class Small_GemmIndexStage : public ::testing::TestWithParam<StageRow>
        {
        };

        TEST_P(Small_GemmIndexStage, StageCoordWalksTheAxisThatIsAdjacent)
        {
            const StageRow row = GetParam();

            expect_coord(Indexing::stage_coord(row.index, row.block.rows, row.block.cols,
                                               row.block.columns_contiguous),
                         row.row, row.col);
        }

        // Index 5 of a 4 x 4 block: down the columns (5 % 4, 5 / 4) or along the rows (5 / 4, 5 % 4).
        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexStage,
                                 ::testing::Values(StageRow{0, {4, 4, 4, true}, 0, 0},
                                                   StageRow{5, {4, 4, 4, true}, 1, 1},
                                                   StageRow{5, {4, 4, 4, false}, 1, 1},
                                                   StageRow{5, {4, 3, 4, true}, 1, 1},
                                                   StageRow{5, {4, 3, 4, false}, 1, 2},
                                                   StageRow{11, {4, 3, 4, true}, 3, 2},
                                                   StageRow{11, {4, 3, 4, false}, 3, 2},
                                                   StageRow{7, {2, 9, 2, true}, 1, 3},
                                                   StageRow{7, {2, 9, 2, false}, 0, 7}));

        class Small_GemmIndexVectorStage : public ::testing::TestWithParam<StageRow>
        {
        };

        TEST_P(Small_GemmIndexVectorStage, StageVectorCoordGivesTheFirstLaneOfTheLoad)
        {
            const StageRow row = GetParam();

            expect_coord(Indexing::stage_vector_coord(row.index, row.block.rows, row.block.cols,
                                                      row.block.lanes,
                                                      row.block.columns_contiguous),
                         row.row, row.col);
        }

        // 8 x 4, 4 lanes, columns adjacent: 2 loads per column; 4 x 8 along rows: 2 loads per row.
        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexVectorStage,
                                 ::testing::Values(StageRow{0, {8, 4, 4, true}, 0, 0},
                                                   StageRow{1, {8, 4, 4, true}, 4, 0},
                                                   StageRow{2, {8, 4, 4, true}, 0, 1},
                                                   StageRow{7, {8, 4, 4, true}, 4, 3},
                                                   StageRow{3, {4, 8, 4, false}, 1, 4},
                                                   StageRow{7, {4, 8, 4, false}, 3, 4},
                                                   StageRow{2, {4, 8, 4, false}, 1, 0}));

        namespace
        {
            /** @brief Holds one group_shape call and the shape its definition gives. */
            struct ShapeRow
            {
                std::size_t budget;
                std::size_t rows;
                std::size_t cols;
            };
        } // namespace

        class Small_GemmIndexShape : public ::testing::TestWithParam<ShapeRow>
        {
        };

        TEST_P(Small_GemmIndexShape, GroupShapeIsTheSquarestPowerOfTwoShapeInTheBudget)
        {
            const ShapeRow row = GetParam();
            const Indexing::GroupShape shape = Indexing::group_shape(row.budget);

            EXPECT_EQ(shape.rows, row.rows);
            EXPECT_EQ(shape.cols, row.cols);
        }

        // The side doubles while its square fits, then a second row of sub-groups is taken if it fits.
        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexShape,
                                 ::testing::Values(ShapeRow{0, 1, 1}, ShapeRow{1, 1, 1},
                                                   ShapeRow{2, 2, 1}, ShapeRow{3, 2, 1},
                                                   ShapeRow{4, 2, 2}, ShapeRow{5, 2, 2},
                                                   ShapeRow{8, 4, 2}, ShapeRow{9, 4, 2},
                                                   ShapeRow{16, 4, 4}, ShapeRow{32, 8, 4},
                                                   ShapeRow{256, 16, 16}, ShapeRow{512, 32, 16}));

        /** @brief Holds a function of three integers and the value its definition gives. */
        struct TripleRow
        {
            int64_t a;
            int64_t b;
            int64_t c;
            int64_t expected;
        };

        class Small_GemmIndexGroupCount : public ::testing::TestWithParam<TripleRow>
        {
        };

        TEST_P(Small_GemmIndexGroupCount, GroupCountCoversTheExtentAndIsAtLeastOne)
        {
            const TripleRow row = GetParam();

            EXPECT_EQ(Indexing::group_count(row.a, row.b), row.expected);
        }

        // ceil(extent / block), floored at 1: 1, 1, 1, 2, 3, 1, 3.
        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexGroupCount,
                                 ::testing::Values(TripleRow{0, 4, 0, 1}, TripleRow{1, 4, 0, 1},
                                                   TripleRow{4, 4, 0, 1}, TripleRow{5, 4, 0, 2},
                                                   TripleRow{9, 4, 0, 3}, TripleRow{9, 16, 0, 1},
                                                   TripleRow{3, 1, 0, 3}));

        class Small_GemmIndexBlockOrigin : public ::testing::TestWithParam<TripleRow>
        {
        };

        TEST_P(Small_GemmIndexBlockOrigin, BlockOriginIsGroupTimesBlockPlusLocal)
        {
            const TripleRow row = GetParam();

            EXPECT_EQ(Indexing::block_origin(row.a, row.b, row.c), row.expected);
        }

        // group * block + local: 0, 64, 35, 13, 5.
        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexBlockOrigin,
                                 ::testing::Values(TripleRow{0, 64, 0, 0}, TripleRow{1, 64, 0, 64},
                                                   TripleRow{2, 16, 3, 35}, TripleRow{3, 4, 1, 13},
                                                   TripleRow{5, 1, 0, 5}));

        class Small_GemmIndexLatticeCoord : public ::testing::TestWithParam<TripleRow>
        {
        };

        TEST_P(Small_GemmIndexLatticeCoord, LatticeCoordIsOriginPlusStepIndexTimesStep)
        {
            const TripleRow row = GetParam();

            EXPECT_EQ(Indexing::lattice_coord(row.a, row.b, row.c), row.expected);
        }

        // origin + step_index * step: 3, 3, 11, 0, 19.
        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexLatticeCoord,
                                 ::testing::Values(TripleRow{3, 0, 8, 3}, TripleRow{3, 5, 0, 3},
                                                   TripleRow{3, 1, 8, 11}, TripleRow{0, 0, 0, 0},
                                                   TripleRow{1, 3, 6, 19}));

        /** @brief Holds one block_is_whole call and the answer. */
        struct WholeRow
        {
            int64_t row0;
            int64_t col0;
            int64_t rows;
            int64_t cols;
            int64_t matrix_rows;
            int64_t matrix_cols;
            bool expected;
        };

        class Small_GemmIndexWhole : public ::testing::TestWithParam<WholeRow>
        {
        };

        TEST_P(Small_GemmIndexWhole, ABlockIsWholeWhenItsFarEdgesAreInsideTheMatrix)
        {
            const WholeRow row = GetParam();

            EXPECT_EQ(Indexing::block_is_whole(row.row0, row.col0, row.rows, row.cols,
                                               row.matrix_rows, row.matrix_cols),
                      row.expected);
        }

        // A block touching the edge is whole; one element past either edge is not; an empty block of an empty matrix is.
        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexWhole,
                                 ::testing::Values(WholeRow{0, 0, 4, 4, 4, 4, true},
                                                   WholeRow{0, 0, 4, 4, 3, 4, false},
                                                   WholeRow{4, 0, 4, 4, 8, 4, true},
                                                   WholeRow{4, 0, 4, 4, 7, 4, false},
                                                   WholeRow{0, 4, 4, 4, 4, 7, false},
                                                   WholeRow{0, 0, 1, 1, 0, 0, false},
                                                   WholeRow{0, 0, 0, 0, 0, 0, true}));

        /** @brief Holds a function of two strides and the value its definition gives. */
        struct StridePairRow
        {
            int64_t row_stride;
            int64_t col_stride;
            int64_t expected;
        };

        class Small_GemmIndexLead : public ::testing::TestWithParam<StridePairRow>
        {
        };

        TEST_P(Small_GemmIndexLead, LeadStrideIsTheStrideOfTheAxisThatIsNotContiguous)
        {
            const StridePairRow row = GetParam();

            EXPECT_EQ(Indexing::lead_stride(row.row_stride, row.col_stride), row.expected);
        }

        // Rows contiguous gives the column stride, columns contiguous the row stride, neither gives 0.
        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexLead,
                                 ::testing::Values(StridePairRow{1, 5, 5}, StridePairRow{5, 1, 5},
                                                   StridePairRow{1, 1, 1}, StridePairRow{3, 4, 0},
                                                   StridePairRow{7, 1, 7}));

        class Small_GemmIndexFragment : public ::testing::TestWithParam<StridePairRow>
        {
        };

        TEST_P(Small_GemmIndexFragment, FragmentStrideStepsByTheAxisThatIsNotContiguous)
        {
            const StridePairRow row = GetParam();

            EXPECT_EQ(Indexing::fragment_stride(row.row_stride, row.col_stride), row.expected);
        }

        // Columns stepping by one give the row stride; otherwise the column stride.
        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexFragment,
                                 ::testing::Values(StridePairRow{7, 1, 7}, StridePairRow{1, 7, 7},
                                                   StridePairRow{1, 1, 1}, StridePairRow{3, 4, 4}));

        /** @brief Holds one strides_are_aligned call and the answer. */
        struct AlignedRow
        {
            int64_t lead;
            int64_t lead_multiple;
            int64_t batch_stride;
            int64_t batch_multiple;
            bool expected;
        };

        class Small_GemmIndexAligned : public ::testing::TestWithParam<AlignedRow>
        {
        };

        TEST_P(Small_GemmIndexAligned, TheLeadMustBePositiveAndBothStridesWholeMultiples)
        {
            const AlignedRow row = GetParam();

            EXPECT_EQ(Indexing::strides_are_aligned(row.lead, row.lead_multiple, row.batch_stride,
                                                    row.batch_multiple),
                      row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexAligned,
                                 ::testing::Values(AlignedRow{8, 4, 16, 4, true},
                                                   AlignedRow{0, 4, 16, 4, false},
                                                   AlignedRow{6, 4, 16, 4, false},
                                                   AlignedRow{8, 4, 18, 4, false},
                                                   AlignedRow{8, 8, 16, 16, true},
                                                   AlignedRow{8, 8, 8, 16, false},
                                                   AlignedRow{4, 4, 0, 4, true}));

        /** @brief Holds one block_fits_int32 call and the answer. */
        struct Int32Row
        {
            int64_t row_stride;
            int64_t col_stride;
            int64_t rows;
            int64_t cols;
            bool expected;
        };

        class Small_GemmIndexInt32 : public ::testing::TestWithParam<Int32Row>
        {
        };

        TEST_P(Small_GemmIndexInt32, ABlockFitsWhenItsLastOffsetIsAtMostInt32Max)
        {
            const Int32Row row = GetParam();

            EXPECT_EQ(Indexing::block_fits_int32(row.row_stride, row.col_stride, row.rows, row.cols),
                      row.expected);
        }

        // 2047 * 2^20 = 2146435072 is under 2^31 - 1, 2048 * 2^20 = 2^31 is over.
        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexInt32,
                                 ::testing::Values(Int32Row{1, 64, 64, 64, true},
                                                   Int32Row{0, 1, 4, 4, false},
                                                   Int32Row{1, 0, 4, 4, false},
                                                   Int32Row{1, 1048576, 1, 2048, true},
                                                   Int32Row{1, 1048576, 1, 2049, false},
                                                   Int32Row{1, 1, 1, 2147483648, true},
                                                   Int32Row{1, 1, 1, 2147483649, false}));

        /** @brief Holds one narrow_offset call and the offset its definition gives. */
        struct NarrowRow
        {
            int r;
            int c;
            int64_t row_stride;
            int64_t col_stride;
            int32_t expected;
        };

        class Small_GemmIndexNarrow : public ::testing::TestWithParam<NarrowRow>
        {
        };

        TEST_P(Small_GemmIndexNarrow, NarrowOffsetIsRowTimesRowStridePlusColumnTimesColumnStride)
        {
            const NarrowRow row = GetParam();

            EXPECT_EQ(Indexing::narrow_offset(row.r, row.c, row.row_stride, row.col_stride),
                      row.expected);
        }

        // 0, 2 * 10 + 3 * 1, 5 * 1 + 1 * 7, 1 * 1 + 1 * 1.
        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexNarrow,
                                 ::testing::Values(NarrowRow{0, 0, 9, 9, 0}, NarrowRow{2, 3, 10, 1, 23},
                                                   NarrowRow{5, 1, 1, 7, 12}, NarrowRow{1, 1, 1, 1, 2}));

        /** @brief Holds one split_group call and the batch entry and block it gives. */
        struct SplitGroupRow
        {
            int64_t group;
            int64_t blocks;
            int64_t batch;
            int64_t block;
        };

        class Small_GemmIndexSplitGroup : public ::testing::TestWithParam<SplitGroupRow>
        {
        };

        TEST_P(Small_GemmIndexSplitGroup, SplitGroupIsTheQuotientAndRemainderByTheBlockCount)
        {
            const SplitGroupRow row = GetParam();
            const Indexing::SplitGroup at = Indexing::split_group(row.group, row.blocks);

            EXPECT_EQ(at.batch, row.batch);
            EXPECT_EQ(at.block, row.block);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexSplitGroup,
                                 ::testing::Values(SplitGroupRow{0, 1, 0, 0}, SplitGroupRow{5, 1, 5, 0},
                                                   SplitGroupRow{5, 3, 1, 2}, SplitGroupRow{8, 4, 2, 0},
                                                   SplitGroupRow{0, 9, 0, 0}, SplitGroupRow{26, 9, 2, 8}));

        /** @brief Holds one split_lane call and the column lane and depth lane it gives. */
        struct SplitLaneRow
        {
            std::size_t lane;
            std::size_t column_lanes;
            std::size_t column;
            std::size_t depth;
        };

        class Small_GemmIndexSplitLane : public ::testing::TestWithParam<SplitLaneRow>
        {
        };

        TEST_P(Small_GemmIndexSplitLane, NeighbouringLanesTakeNeighbouringColumnsThenKSteps)
        {
            const SplitLaneRow row = GetParam();
            const Indexing::SplitLane at = Indexing::split_lane(row.lane, row.column_lanes);

            EXPECT_EQ(at.column, row.column);
            EXPECT_EQ(at.depth, row.depth);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexSplitLane,
                                 ::testing::Values(SplitLaneRow{0, 1, 0, 0}, SplitLaneRow{5, 1, 0, 5},
                                                   SplitLaneRow{5, 4, 1, 1}, SplitLaneRow{7, 2, 1, 3},
                                                   SplitLaneRow{15, 16, 15, 0},
                                                   SplitLaneRow{16, 16, 0, 1}));

        /** @brief Holds one split_origin call and the first row and column it gives. */
        struct SplitOriginRow
        {
            int64_t block;
            int64_t n_blocks;
            int64_t item_rows;
            int64_t block_cols;
            std::size_t column_lane;
            int64_t item_cols;
            int64_t row;
            int64_t col;
        };

        class Small_GemmIndexSplitOrigin : public ::testing::TestWithParam<SplitOriginRow>
        {
        };

        TEST_P(Small_GemmIndexSplitOrigin, TheBlockPicksTheRowBandAndTheColumnBandAndTheLaneTheColumns)
        {
            const SplitOriginRow row = GetParam();

            expect_coord(Indexing::split_origin(row.block, row.n_blocks, row.item_rows,
                                                row.block_cols, row.column_lane, row.item_cols),
                         row.row, row.col);
        }

        // Block 3 of 2 column bands: row 4, column 8 + 1 * 2. Block 5 of 3: row 4, column 2 * 12 + 2 * 2.
        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexSplitOrigin,
                                 ::testing::Values(SplitOriginRow{0, 1, 4, 16, 0, 4, 0, 0},
                                                   SplitOriginRow{3, 2, 4, 8, 1, 2, 4, 10},
                                                   SplitOriginRow{5, 3, 4, 12, 2, 2, 4, 28},
                                                   SplitOriginRow{2, 1, 2, 4, 0, 2, 4, 0}));

        /** @brief Holds a function of three sizes and the value its definition gives. */
        struct SizeTripleRow
        {
            std::size_t a;
            std::size_t b;
            std::size_t c;
            std::size_t expected;
        };

        class Small_GemmIndexFold : public ::testing::TestWithParam<SizeTripleRow>
        {
        };

        TEST_P(Small_GemmIndexFold, FoldPartnerIsTheLaneAtOffDepthLanesFurther)
        {
            const SizeTripleRow row = GetParam();

            EXPECT_EQ(Indexing::fold_partner(row.a, row.b, row.c), row.expected);
        }

        // lane + off * column_lanes: 1, 11, 4, 7.
        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexFold,
                                 ::testing::Values(SizeTripleRow{0, 1, 1, 1}, SizeTripleRow{3, 2, 4, 11},
                                                   SizeTripleRow{0, 4, 1, 4}, SizeTripleRow{5, 1, 2, 7}));

        class Small_GemmIndexPartial : public ::testing::TestWithParam<SizeTripleRow>
        {
        };

        TEST_P(Small_GemmIndexPartial, PartialSlotIsElementTimesWidthPlusLane)
        {
            const SizeTripleRow row = GetParam();

            EXPECT_EQ(Indexing::partial_slot(row.a, row.b, row.c), row.expected);
        }

        // element * width + lane: 5, 37, 48, 0.
        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexPartial,
                                 ::testing::Values(SizeTripleRow{0, 32, 5, 5}, SizeTripleRow{1, 32, 5, 37},
                                                   SizeTripleRow{3, 16, 0, 48}, SizeTripleRow{0, 1, 0, 0}));

        /** @brief Holds one sub_group_origin call on the 64 x 64 work-group of 2 x 2 sub-groups of 32. */
        struct SubGroupRow
        {
            int64_t group_row;
            int64_t group_col;
            int64_t sub;
            int64_t row;
            int64_t col;
        };

        class Small_GemmIndexSubGroup : public ::testing::TestWithParam<SubGroupRow>
        {
        };

        TEST_P(Small_GemmIndexSubGroup, TheSubGroupsTileTheWorkGroupBlockRowFirst)
        {
            const SubGroupRow row = GetParam();

            expect_coord(Indexing::sub_group_origin(row.group_row, row.group_col, row.sub, 64, 32, 2),
                         row.row, row.col);
        }

        // Sub-group 1 is the top right quadrant, 2 the bottom left, 3 the bottom right; group (1, 2) starts at (64, 128).
        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexSubGroup,
                                 ::testing::Values(SubGroupRow{0, 0, 0, 0, 0}, SubGroupRow{0, 0, 1, 0, 32},
                                                   SubGroupRow{0, 0, 2, 32, 0}, SubGroupRow{0, 0, 3, 32, 32},
                                                   SubGroupRow{1, 2, 3, 96, 160},
                                                   SubGroupRow{2, 0, 2, 160, 0}));

        /** @brief Holds one fragment_inside call on fragments of 16 and the answer. */
        struct FragmentRow
        {
            int64_t origin;
            int64_t f;
            int64_t extent;
            bool expected;
        };

        class Small_GemmIndexFragmentInside : public ::testing::TestWithParam<FragmentRow>
        {
        };

        TEST_P(Small_GemmIndexFragmentInside, AFragmentIsInsideWhenItsFirstRowOrColumnIsInTheExtent)
        {
            const FragmentRow row = GetParam();

            EXPECT_EQ(Indexing::fragment_inside(row.origin, row.f, 16, row.extent), row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexFragmentInside,
                                 ::testing::Values(FragmentRow{0, 0, 16, true}, FragmentRow{0, 1, 16, false},
                                                   FragmentRow{0, 1, 17, true}, FragmentRow{32, 1, 48, false},
                                                   FragmentRow{32, 1, 49, true}, FragmentRow{0, 0, 0, false}));

        /** @brief Holds one fragment_element call on a fragment of 16 and the coordinate it gives. */
        struct FragmentElementRow
        {
            int64_t e;
            int64_t row;
            int64_t col;
        };

        class Small_GemmIndexFragmentElement : public ::testing::TestWithParam<FragmentElementRow>
        {
        };

        TEST_P(Small_GemmIndexFragmentElement, ElementsOfARowMajorFragmentRunAlongTheRows)
        {
            const FragmentElementRow row = GetParam();

            expect_coord(Indexing::fragment_element(row.e, 16), row.row, row.col);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GemmIndexFragmentElement,
                                 ::testing::Values(FragmentElementRow{0, 0, 0}, FragmentElementRow{15, 0, 15},
                                                   FragmentElementRow{16, 1, 0}, FragmentElementRow{37, 2, 5},
                                                   FragmentElementRow{255, 15, 15}));
    } // namespace Test
} // namespace SushiBLAS
