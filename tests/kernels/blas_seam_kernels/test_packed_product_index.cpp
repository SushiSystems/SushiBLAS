/**************************************************************************/
/* test_packed_product_index.cpp                                          */
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
 * @file test_packed_product_index.cpp
 * @brief Tests the macro-block and panel index math of the packed product against hand values and a distinct-origin check.
 * @author Mustafa Garip
 */

#include <set>
#include <tuple>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_seam_kernels/packed_product_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Indexing = Kernels::Indexing;

        namespace
        {
            /** @brief Holds the macro-block grid of a packed product. */
            struct BlockGrid
            {
                int64_t row_blocks;
                int64_t col_blocks;
                int64_t batch;
                int64_t block_rows;
                int64_t block_cols;
            };

            /** @brief Checks that the macro-block ids name every (batch, row0, col0) of the grid exactly once. */
            void expect_blocks_name_every_origin_once(const BlockGrid& grid)
            {
                std::set<std::tuple<int64_t, int64_t, int64_t>> seen;
                const int64_t blocks = grid.row_blocks * grid.col_blocks * grid.batch;
                for (int64_t block = 0; block < blocks; ++block)
                {
                    const Indexing::PackedBlock at = Indexing::packed_block(
                        block, grid.row_blocks, grid.col_blocks, grid.block_rows, grid.block_cols);
                    EXPECT_TRUE(seen.emplace(at.batch, at.row0, at.col0).second) << "block " << block;
                }
                EXPECT_EQ(static_cast<int64_t>(seen.size()), blocks);
            }
        } // namespace

        class Small_PackedProductIndexGrid : public ::testing::TestWithParam<BlockGrid>
        {
        };

        TEST_P(Small_PackedProductIndexGrid, EveryMacroBlockIdNamesADistinctOrigin)
        {
            expect_blocks_name_every_origin_once(GetParam());
        }

        // One block, one row of blocks, one column of blocks, a 2 x 3 grid, batches of 1, 2, 3 and 5.
        INSTANTIATE_TEST_SUITE_P(, Small_PackedProductIndexGrid,
                                 ::testing::Values(BlockGrid{1, 1, 1, 6, 16}, BlockGrid{1, 5, 1, 6, 16},
                                                   BlockGrid{3, 1, 2, 6, 16}, BlockGrid{2, 3, 1, 6, 16},
                                                   BlockGrid{2, 3, 3, 4, 8}, BlockGrid{1, 1, 5, 4, 8},
                                                   BlockGrid{3, 3, 2, 12, 16}));

        /** @brief Holds one packed_block call and the origin its definition gives. */
        struct PackedBlockRow
        {
            int64_t block;
            BlockGrid grid;
            Indexing::PackedBlock expected;
        };

        class Small_PackedProductIndexBlock : public ::testing::TestWithParam<PackedBlockRow>
        {
        };

        TEST_P(Small_PackedProductIndexBlock, RowsAdvanceFastestInsideAMatrixAndTheBatchSlowest)
        {
            const PackedBlockRow row = GetParam();
            const Indexing::PackedBlock at =
                Indexing::packed_block(row.block, row.grid.row_blocks, row.grid.col_blocks,
                                       row.grid.block_rows, row.grid.block_cols);

            EXPECT_EQ(at.batch, row.expected.batch);
            EXPECT_EQ(at.row0, row.expected.row0);
            EXPECT_EQ(at.col0, row.expected.col0);
        }

        // A 2 x 3 grid holds 6 blocks per matrix: id 6 opens the second matrix, id 13 is row block 1 of the third.
        INSTANTIATE_TEST_SUITE_P(
            , Small_PackedProductIndexBlock,
            ::testing::Values(PackedBlockRow{0, BlockGrid{1, 1, 1, 6, 16}, {0, 0, 0}},
                              PackedBlockRow{1, BlockGrid{1, 1, 2, 6, 16}, {1, 0, 0}},
                              PackedBlockRow{1, BlockGrid{2, 3, 3, 6, 16}, {0, 6, 0}},
                              PackedBlockRow{2, BlockGrid{2, 3, 3, 6, 16}, {0, 0, 16}},
                              PackedBlockRow{5, BlockGrid{2, 3, 3, 6, 16}, {0, 6, 32}},
                              PackedBlockRow{6, BlockGrid{2, 3, 3, 6, 16}, {1, 0, 0}},
                              PackedBlockRow{13, BlockGrid{2, 3, 3, 6, 16}, {2, 6, 0}}));

        /** @brief Holds one micro_tile_index call and the index its definition gives. */
        struct MicroTileRow
        {
            int64_t g;
            int64_t h;
            int64_t row_groups;
            int64_t expected;
        };

        class Small_PackedProductIndexTile : public ::testing::TestWithParam<MicroTileRow>
        {
        };

        TEST_P(Small_PackedProductIndexTile, MicroTileIndexCountsRowGroupsFastest)
        {
            const MicroTileRow row = GetParam();

            EXPECT_EQ(Indexing::micro_tile_index(row.g, row.h, row.row_groups), row.expected);
        }

        // h * row_groups + g: 0, 2, 3, 8, 5.
        INSTANTIATE_TEST_SUITE_P(, Small_PackedProductIndexTile,
                                 ::testing::Values(MicroTileRow{0, 0, 3, 0}, MicroTileRow{2, 0, 3, 2},
                                                   MicroTileRow{0, 1, 3, 3}, MicroTileRow{2, 2, 3, 8},
                                                   MicroTileRow{1, 4, 1, 5}));

        /** @brief Holds one panel offset call: a unit index, the reals one unit holds and the offset. */
        struct PanelOffsetRow
        {
            int64_t index;
            int64_t unit_reals;
            int64_t expected;
        };

        class Small_PackedProductIndexPanel : public ::testing::TestWithParam<PanelOffsetRow>
        {
        };

        TEST_P(Small_PackedProductIndexPanel, TheMicroPanelStartsAfterWholePrecedingMicroPanels)
        {
            const PanelOffsetRow row = GetParam();

            EXPECT_EQ(Indexing::micro_panel_offset(row.index, row.unit_reals), row.expected);
        }

        TEST_P(Small_PackedProductIndexPanel, TheSliceOfATermStartsAfterWholePrecedingSlices)
        {
            const PanelOffsetRow row = GetParam();

            EXPECT_EQ(Indexing::slice_offset(row.index, row.unit_reals), row.expected);
        }

        // index * unit_reals: 0, 0, 24, 96, 9.
        INSTANTIATE_TEST_SUITE_P(, Small_PackedProductIndexPanel,
                                 ::testing::Values(PanelOffsetRow{0, 24, 0}, PanelOffsetRow{5, 0, 0},
                                                   PanelOffsetRow{1, 24, 24}, PanelOffsetRow{4, 24, 96},
                                                   PanelOffsetRow{9, 1, 9}));
    } // namespace Test
} // namespace SushiBLAS
