/**************************************************************************/
/* test_window_index.cpp                                                  */
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
 * @file test_window_index.cpp
 * @brief Tests the sliding-window index math against a search over the windows, nested loops and hand values.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/window_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Indexing = Kernels::Indexing;

        /** @brief Holds one axis of a window: the input extent, the window and its padding. */
        struct AxisRow
        {
            int64_t in_extent;
            int64_t kernel;
            int64_t stride;
            int64_t dilation;
            int64_t pad_before;
            int64_t pad_after;
        };

        /** @brief Returns the output position whose window reads in_pos through tap, or -1, by searching every window. */
        static int64_t scattered_by_search(const AxisRow& axis, int64_t in_pos, int64_t tap,
                                           int64_t out_extent)
        {
            for (int64_t out_pos = 0; out_pos < out_extent; ++out_pos)
            {
                if (Indexing::gather_coord(out_pos, axis.stride, axis.pad_before, tap,
                                           axis.dilation) == in_pos)
                    return out_pos;
            }
            return -1;
        }

        /** @brief Checks scatter_coord against the search for every input position and tap of an axis. */
        static void expect_scatter_matches_the_search(const AxisRow& axis)
        {
            const int64_t out_extent = Spatial::output_extent(
                axis.in_extent, axis.pad_before, axis.pad_after, axis.dilation, axis.kernel,
                axis.stride);
            for (int64_t in_pos = 0; in_pos < axis.in_extent; ++in_pos)
            {
                for (int64_t tap = 0; tap < axis.kernel; ++tap)
                {
                    EXPECT_EQ(Indexing::scatter_coord(in_pos, axis.stride, axis.pad_before, tap,
                                                      axis.dilation, out_extent),
                              scattered_by_search(axis, in_pos, tap, out_extent))
                        << "in_pos " << in_pos << " tap " << tap;
                }
            }
        }

        class Small_WindowIndexAxes : public ::testing::TestWithParam<AxisRow>
        {
        };

        TEST_P(Small_WindowIndexAxes, ScatterCoordFindsTheWindowThatReadsAPosition)
        {
            expect_scatter_matches_the_search(GetParam());
        }

        // Extents {0, 1, 2, 3, 5, 9}; a padded and an unpadded edge; stride and dilation above 1.
        INSTANTIATE_TEST_SUITE_P(, Small_WindowIndexAxes,
                                 ::testing::Values(AxisRow{0, 1, 1, 1, 0, 0},
                                                   AxisRow{1, 1, 1, 1, 0, 0},
                                                   AxisRow{1, 3, 1, 1, 1, 1},
                                                   AxisRow{2, 2, 2, 1, 0, 0},
                                                   AxisRow{3, 3, 1, 1, 1, 1},
                                                   AxisRow{3, 2, 1, 2, 2, 0},
                                                   AxisRow{5, 3, 2, 1, 0, 0},
                                                   AxisRow{5, 3, 2, 2, 2, 2},
                                                   AxisRow{9, 3, 3, 1, 1, 0},
                                                   AxisRow{9, 2, 2, 3, 0, 3},
                                                   AxisRow{9, 1, 4, 1, 0, 0}));

        /** @brief Holds one gather_coord call and the value its formula gives. */
        struct GatherRow
        {
            int64_t out_pos;
            int64_t stride;
            int64_t pad_before;
            int64_t tap;
            int64_t dilation;
            int64_t expected;
        };

        class Small_WindowIndexGather : public ::testing::TestWithParam<GatherRow>
        {
        };

        TEST_P(Small_WindowIndexGather, GatherCoordIsPositionTimesStrideMinusPadPlusTapTimesDilation)
        {
            const GatherRow row = GetParam();

            EXPECT_EQ(Indexing::gather_coord(row.out_pos, row.stride, row.pad_before, row.tap,
                                             row.dilation),
                      row.expected);
        }

        // out * stride - pad + tap * dilation: 0-1+0, 0-0+0, 4-1+2, 12-2+6, 1-0+4, 0-2+3.
        INSTANTIATE_TEST_SUITE_P(, Small_WindowIndexGather,
                                 ::testing::Values(GatherRow{0, 1, 1, 0, 1, -1},
                                                   GatherRow{0, 1, 0, 0, 1, 0},
                                                   GatherRow{2, 2, 1, 1, 2, 5},
                                                   GatherRow{4, 3, 2, 2, 3, 16},
                                                   GatherRow{1, 1, 0, 2, 2, 5},
                                                   GatherRow{0, 2, 2, 3, 1, 1}));

        /** @brief Holds one in_extent call. */
        struct ExtentRow
        {
            int64_t pos;
            int64_t extent;
            bool expected;
        };

        class Small_WindowIndexExtent : public ::testing::TestWithParam<ExtentRow>
        {
        };

        TEST_P(Small_WindowIndexExtent, InExtentIsTrueOnlyInsideTheHalfOpenRange)
        {
            const ExtentRow row = GetParam();

            EXPECT_EQ(Indexing::in_extent(row.pos, row.extent), row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_WindowIndexExtent,
                                 ::testing::Values(ExtentRow{-1, 3, false}, ExtentRow{0, 3, true},
                                                   ExtentRow{2, 3, true}, ExtentRow{3, 3, false},
                                                   ExtentRow{0, 0, false}, ExtentRow{0, 1, true},
                                                   ExtentRow{1, 1, false}));

        /** @brief Holds the extents of the windows of one batch. */
        struct WindowGridRow
        {
            int64_t batch;
            int64_t out_h;
            int64_t out_w;
        };

        /** @brief Returns the windows in storage order, listed by nested loops. */
        static std::vector<Indexing::WindowCoord> listed_windows(const WindowGridRow& grid)
        {
            std::vector<Indexing::WindowCoord> windows;
            for (int64_t n = 0; n < grid.batch; ++n)
            {
                for (int64_t h_o = 0; h_o < grid.out_h; ++h_o)
                {
                    for (int64_t w_o = 0; w_o < grid.out_w; ++w_o)
                        windows.push_back({n, h_o, w_o});
                }
            }
            return windows;
        }

        /** @brief Checks window_coord and window_row against the listing, both ways. */
        static void expect_windows_follow_the_listing(const WindowGridRow& grid)
        {
            const std::vector<Indexing::WindowCoord> listed = listed_windows(grid);
            for (std::size_t i = 0; i < listed.size(); ++i)
            {
                const Indexing::WindowCoord at =
                    Indexing::window_coord(static_cast<int64_t>(i), grid.out_h, grid.out_w);
                EXPECT_EQ(at.n, listed[i].n) << "flat " << i;
                EXPECT_EQ(at.h_o, listed[i].h_o) << "flat " << i;
                EXPECT_EQ(at.w_o, listed[i].w_o) << "flat " << i;
                EXPECT_EQ(Indexing::window_row(listed[i], grid.out_h, grid.out_w),
                          static_cast<int64_t>(i));
            }
        }

        class Small_WindowIndexGrids : public ::testing::TestWithParam<WindowGridRow>
        {
        };

        TEST_P(Small_WindowIndexGrids, WindowCoordAndWindowRowFollowTheNestedLoopOrder)
        {
            expect_windows_follow_the_listing(GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_WindowIndexGrids,
                                 ::testing::Values(WindowGridRow{0, 3, 3}, WindowGridRow{2, 0, 3},
                                                   WindowGridRow{1, 1, 1}, WindowGridRow{2, 3, 5},
                                                   WindowGridRow{3, 5, 9}, WindowGridRow{1, 9, 2},
                                                   WindowGridRow{5, 2, 1}));

        /** @brief Holds the extents of the patch of one window. */
        struct PatchRow
        {
            int64_t kernel_h;
            int64_t kernel_w;
            int64_t channels;
        };

        /** @brief Returns the patch elements in storage order, listed by nested loops. */
        static std::vector<Indexing::TapCoord> listed_taps(const PatchRow& patch)
        {
            std::vector<Indexing::TapCoord> taps;
            for (int64_t r = 0; r < patch.kernel_h; ++r)
            {
                for (int64_t s = 0; s < patch.kernel_w; ++s)
                {
                    for (int64_t c = 0; c < patch.channels; ++c)
                        taps.push_back({r, s, c});
                }
            }
            return taps;
        }

        /** @brief Checks tap_coord and patch_column against the listing, both ways. */
        static void expect_taps_follow_the_listing(const PatchRow& patch)
        {
            const std::vector<Indexing::TapCoord> listed = listed_taps(patch);
            for (std::size_t i = 0; i < listed.size(); ++i)
            {
                const Indexing::TapCoord at = Indexing::tap_coord(
                    static_cast<int64_t>(i), patch.channels, patch.kernel_w);
                EXPECT_EQ(at.r, listed[i].r) << "flat " << i;
                EXPECT_EQ(at.s, listed[i].s) << "flat " << i;
                EXPECT_EQ(at.c, listed[i].c) << "flat " << i;
                EXPECT_EQ(Indexing::patch_column(listed[i], patch.channels, patch.kernel_w),
                          static_cast<int64_t>(i));
            }
        }

        class Small_WindowIndexPatches : public ::testing::TestWithParam<PatchRow>
        {
        };

        TEST_P(Small_WindowIndexPatches, TapCoordAndPatchColumnFollowTheNestedLoopOrder)
        {
            expect_taps_follow_the_listing(GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_WindowIndexPatches,
                                 ::testing::Values(PatchRow{0, 3, 3}, PatchRow{1, 1, 1},
                                                   PatchRow{2, 3, 5}, PatchRow{3, 2, 9},
                                                   PatchRow{5, 1, 2}, PatchRow{1, 9, 3}));

        TEST(Small_WindowIndex, WindowOriginIsTapTimesDilationMinusPad)
        {
            // 2 * 3 - 1 = 5.
            EXPECT_EQ(Indexing::window_origin(2, 3, 1), 5);
        }

        TEST(Small_WindowIndex, OriginCoordIsPositionTimesStridePlusOrigin)
        {
            // 4 * 2 + (-1) = 7.
            EXPECT_EQ(Indexing::origin_coord(4, 2, -1), 7);
        }

        TEST(Small_WindowIndex, WindowAreaIsKernelHeightTimesWidth)
        {
            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 2;

            EXPECT_EQ(Indexing::window_area(g), 6);
        }

        TEST(Small_WindowIndex, ColumnOffsetIsRowTimesRowStrideAndColTimesElementStride)
        {
            Spatial::ColumnSpan cols;
            cols.row_stride = 20;
            cols.element_stride = 2;

            // 3 * 20 + 4 * 2 = 68.
            EXPECT_EQ(Indexing::column_offset(cols, 3, 4), 68);
        }

        TEST(Small_WindowIndex, TapOffsetIsTheWeightedSumOfTheThreeTapCoordinates)
        {
            Spatial::TapSpan taps;
            taps.row_stride = 100;
            taps.col_stride = 10;
            taps.channel_stride = 3;

            // 1 * 100 + 2 * 10 + 4 * 3 = 132.
            EXPECT_EQ(Indexing::tap_offset(taps, Indexing::TapCoord{1, 2, 4}), 132);
        }
    }
}
