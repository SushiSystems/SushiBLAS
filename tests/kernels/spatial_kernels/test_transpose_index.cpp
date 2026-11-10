/**************************************************************************/
/* test_transpose_index.cpp                                               */
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
 * @file test_transpose_index.cpp
 * @brief Tests the tile geometry and index math of the transposing layout copy, including a host run of the tiled copy.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/pixel_index.hpp>
#include <SushiBLAS/spatial_kernels/transpose_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Indexing = Kernels::Indexing;

        /** @brief Holds one tile_count call. */
        struct TileCountRow
        {
            int64_t extent;
            int64_t expected;
        };

        class Small_TransposeIndexTileCount : public ::testing::TestWithParam<TileCountRow>
        {
        };

        TEST_P(Small_TransposeIndexTileCount, TileCountRoundsTheExtentUpToWholeTiles)
        {
            const TileCountRow row = GetParam();

            EXPECT_EQ(Indexing::tile_count(row.extent), row.expected);
        }

        // Tile side 32: 0 needs none, 1..32 need one, 33..64 need two, 65 needs three.
        INSTANTIATE_TEST_SUITE_P(, Small_TransposeIndexTileCount,
                                 ::testing::Values(TileCountRow{0, 0}, TileCountRow{1, 1},
                                                   TileCountRow{2, 1}, TileCountRow{3, 1},
                                                   TileCountRow{5, 1}, TileCountRow{9, 1},
                                                   TileCountRow{31, 1}, TileCountRow{32, 1},
                                                   TileCountRow{33, 2}, TileCountRow{64, 2},
                                                   TileCountRow{65, 3}));

        namespace
        {
            /** @brief Holds one tile_slot call. */
            struct SlotRow
            {
                int64_t row;
                int64_t col;
                int64_t expected;
            };
        } // namespace

        class Small_TransposeIndexSlot : public ::testing::TestWithParam<SlotRow>
        {
        };

        TEST_P(Small_TransposeIndexSlot, TileSlotUsesAPaddedRowOf33)
        {
            const SlotRow row = GetParam();

            EXPECT_EQ(Indexing::tile_slot(row.row, row.col), row.expected);
        }

        // row * 33 + col: 0, 33, 2 * 33 + 5, 31 * 33 + 31.
        INSTANTIATE_TEST_SUITE_P(, Small_TransposeIndexSlot,
                                 ::testing::Values(SlotRow{0, 0, 0}, SlotRow{1, 0, 33},
                                                   SlotRow{2, 5, 71}, SlotRow{31, 31, 1054}));

        TEST(Small_TransposeIndex, TileElementsIs32Times33)
        {
            EXPECT_EQ(Indexing::tile_elements(), 1056);
        }

        /** @brief Holds one tile_live call on a plan of 3 slow by 5 fast. */
        struct LiveRow
        {
            int64_t slow;
            int64_t fast;
            bool expected;
        };

        class Small_TransposeIndexLive : public ::testing::TestWithParam<LiveRow>
        {
        };

        TEST_P(Small_TransposeIndexLive, TileLiveIsTrueOnlyInsideBothExtents)
        {
            Indexing::TransposePlan plan;
            plan.slow = 3;
            plan.fast = 5;
            const LiveRow row = GetParam();

            EXPECT_EQ(Indexing::tile_live(plan, row.slow, row.fast), row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_TransposeIndexLive,
                                 ::testing::Values(LiveRow{0, 0, true}, LiveRow{2, 4, true},
                                                   LiveRow{3, 4, false}, LiveRow{2, 5, false},
                                                   LiveRow{3, 5, false}));

        TEST(Small_TransposeIndex, ReadAndWriteOffsetsUseTheSlowInputStepAndTheFastOutputStep)
        {
            Indexing::TransposePlan plan;
            plan.in_slow = 7;
            plan.out_fast = 11;

            // read: 2 * 7 + 3 = 17; write: 2 + 3 * 11 = 35.
            EXPECT_EQ(Indexing::transpose_read_offset(plan, 2, 3), 17);
            EXPECT_EQ(Indexing::transpose_write_offset(plan, 2, 3), 35);
        }

        /** @brief Returns the span of a batch of 2 images of height x width x channels, NHWC to NCHW or back. */
        static Spatial::WindowSpan converting_span(int64_t height, int64_t width, int64_t channels,
                                                   bool to_nchw)
        {
            Spatial::WindowSpan span;
            span.batch = 2;
            span.in_h = height;
            span.in_w = width;
            span.channels = channels;
            span.out_h = height;
            span.out_w = width;
            const int64_t pixels = height * width;
            const int64_t nhwc_batch = pixels * channels;
            const Indexing::Strides nhwc{nhwc_batch, width * channels, channels, 1};
            const Indexing::Strides nchw{nhwc_batch, width, 1, pixels};
            const Indexing::Strides& in = to_nchw ? nhwc : nchw;
            const Indexing::Strides& out = to_nchw ? nchw : nhwc;
            span.in_batch_stride = in.batch;
            span.in_row_stride = in.row;
            span.in_col_stride = in.col;
            span.in_channel_stride = in.channel;
            span.out_batch_stride = out.batch;
            span.out_row_stride = out.row;
            span.out_col_stride = out.col;
            span.out_channel_stride = out.channel;
            return span;
        }

        /** @brief Holds one layout conversion: the image extents and its direction. */
        struct ConvertRow
        {
            int64_t height;
            int64_t width;
            int64_t channels;
            bool to_nchw;
        };

        /** @brief Runs the tiled copy of the SYCL kernel on the host, one work-item at a time. */
        static std::vector<int64_t> tiled_copy_on_the_host(const Indexing::TransposePlan& plan,
                                                           int64_t batch,
                                                           const std::vector<int64_t>& in,
                                                           std::size_t out_size)
        {
            constexpr int K_TILE = Indexing::K_TRANSPOSE_TILE;
            constexpr int K_ROWS = Indexing::K_TRANSPOSE_ROWS;
            std::vector<int64_t> out(out_size, -1);
            std::vector<int64_t> tile(static_cast<std::size_t>(Indexing::tile_elements()), -1);
            for (int64_t n = 0; n < batch; ++n)
            {
                for (int64_t gs = 0; gs < Indexing::tile_count(plan.slow); ++gs)
                {
                    for (int64_t gf = 0; gf < Indexing::tile_count(plan.fast); ++gf)
                    {
                        for (int ty = 0; ty < K_ROWS; ++ty)
                        {
                            for (int tx = 0; tx < K_TILE; ++tx)
                            {
                                for (int r = ty; r < K_TILE; r += K_ROWS)
                                {
                                    const int64_t s = gs * K_TILE + r;
                                    const int64_t f = gf * K_TILE + tx;
                                    if (Indexing::tile_live(plan, s, f))
                                    {
                                        tile[static_cast<std::size_t>(Indexing::tile_slot(r, tx))] =
                                            in[static_cast<std::size_t>(
                                                n * plan.in_batch +
                                                Indexing::transpose_read_offset(plan, s, f))];
                                    }
                                }
                            }
                        }
                        for (int ty = 0; ty < K_ROWS; ++ty)
                        {
                            for (int tx = 0; tx < K_TILE; ++tx)
                            {
                                for (int r = ty; r < K_TILE; r += K_ROWS)
                                {
                                    const int64_t s = gs * K_TILE + tx;
                                    const int64_t f = gf * K_TILE + r;
                                    if (Indexing::tile_live(plan, s, f))
                                    {
                                        out[static_cast<std::size_t>(
                                            n * plan.out_batch +
                                            Indexing::transpose_write_offset(plan, s, f))] =
                                            tile[static_cast<std::size_t>(
                                                Indexing::tile_slot(tx, r))];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return out;
        }

        /** @brief Checks that the tiled copy moves each logical element to its place in the output layout. */
        static void expect_tiled_copy_moves_every_element(const ConvertRow& row)
        {
            const Spatial::WindowSpan span =
                converting_span(row.height, row.width, row.channels, row.to_nchw);
            Indexing::TransposePlan plan;
            ASSERT_TRUE(Indexing::transposes(span, plan));
            const std::size_t size = static_cast<std::size_t>(2 * row.height * row.width * row.channels);
            std::vector<int64_t> in(size);
            for (std::size_t i = 0; i < size; ++i)
                in[i] = static_cast<int64_t>(i);

            const std::vector<int64_t> out = tiled_copy_on_the_host(plan, span.batch, in, size);

            const Indexing::Strides in_strides = Indexing::input_strides(span);
            const Indexing::Strides out_strides = Indexing::output_strides(span);
            for (int64_t n = 0; n < span.batch; ++n)
            {
                for (int64_t h = 0; h < row.height; ++h)
                {
                    for (int64_t w = 0; w < row.width; ++w)
                    {
                        for (int64_t c = 0; c < row.channels; ++c)
                        {
                            EXPECT_EQ(out[static_cast<std::size_t>(out_strides.offset(n, h, w, c))],
                                      in[static_cast<std::size_t>(in_strides.offset(n, h, w, c))])
                                << "(" << n << ", " << h << ", " << w << ", " << c << ")";
                        }
                    }
                }
            }
        }

        class Small_TransposeIndexConversions : public ::testing::TestWithParam<ConvertRow>
        {
        };

        TEST_P(Small_TransposeIndexConversions, TheTiledCopyMovesEveryElementToItsPlace)
        {
            expect_tiled_copy_moves_every_element(GetParam());
        }

        // Tile side 32: axes of 1, 2, 3, 5, 9 and of 31, 32, 33 elements, so the last tile is partial.
        INSTANTIATE_TEST_SUITE_P(, Small_TransposeIndexConversions,
                                 ::testing::Values(ConvertRow{1, 1, 1, true},
                                                   ConvertRow{2, 3, 4, true},
                                                   ConvertRow{3, 3, 5, false},
                                                   ConvertRow{1, 9, 2, true},
                                                   ConvertRow{1, 31, 33, true},
                                                   ConvertRow{1, 32, 32, false},
                                                   ConvertRow{1, 33, 31, true},
                                                   ConvertRow{3, 11, 5, false},
                                                   ConvertRow{1, 5, 65, true}));

        namespace
        {
            /** @brief Holds one transposes call: the span and the plan it must give, or none. */
            struct PlanRow
            {
                bool to_nchw;
                int64_t fast;
                int64_t slow;
                int64_t in_slow;
                int64_t out_fast;
            };
        } // namespace

        class Small_TransposeIndexPlans : public ::testing::TestWithParam<PlanRow>
        {
        };

        TEST_P(Small_TransposeIndexPlans, TransposesFillsThePlanOfEachDirection)
        {
            const PlanRow row = GetParam();
            const Spatial::WindowSpan span = converting_span(2, 3, 4, row.to_nchw);
            Indexing::TransposePlan plan;

            ASSERT_TRUE(Indexing::transposes(span, plan));

            EXPECT_EQ(plan.fast, row.fast);
            EXPECT_EQ(plan.slow, row.slow);
            EXPECT_EQ(plan.in_slow, row.in_slow);
            EXPECT_EQ(plan.out_fast, row.out_fast);
            EXPECT_EQ(plan.in_batch, 24);
            EXPECT_EQ(plan.out_batch, 24);
        }

        // 6 pixels, 4 channels: NHWC to NCHW is fast 4, slow 6; NCHW to NHWC swaps the extents and steps.
        INSTANTIATE_TEST_SUITE_P(, Small_TransposeIndexPlans,
                                 ::testing::Values(PlanRow{true, 4, 6, 4, 6},
                                                   PlanRow{false, 6, 4, 6, 4}));

        TEST(Small_TransposeIndex, TransposesRefusesAnUnchangedLayout)
        {
            Spatial::WindowSpan span = converting_span(2, 3, 4, true);
            span.out_batch_stride = span.in_batch_stride;
            span.out_row_stride = span.in_row_stride;
            span.out_col_stride = span.in_col_stride;
            span.out_channel_stride = span.in_channel_stride;
            Indexing::TransposePlan plan;

            EXPECT_FALSE(Indexing::transposes(span, plan));
        }

        TEST(Small_TransposeIndex, TransposesRefusesAPaddedRowOfPixels)
        {
            Spatial::WindowSpan span = converting_span(2, 3, 4, true);
            span.in_row_stride = 20;
            Indexing::TransposePlan plan;

            EXPECT_FALSE(Indexing::transposes(span, plan));
        }
    }
}
