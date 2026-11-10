/**************************************************************************/
/* test_pixel_index.cpp                                                   */
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
 * @file test_pixel_index.cpp
 * @brief Tests the NHWC flat-index and offset math against nested loops and hand values.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/pixel_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Holds the extents of one tensor. */
            struct ShapeRow
            {
                int64_t batch;
                int64_t height;
                int64_t width;
                int64_t channels;
            };
        } // namespace

        /** @brief Returns the positions of a tensor in storage order, listed by nested loops. */
        static std::vector<Kernels::Indexing::PixelCoord> listed_positions(const ShapeRow& shape)
        {
            std::vector<Kernels::Indexing::PixelCoord> positions;
            for (int64_t n = 0; n < shape.batch; ++n)
            {
                for (int64_t h = 0; h < shape.height; ++h)
                {
                    for (int64_t w = 0; w < shape.width; ++w)
                    {
                        for (int64_t c = 0; c < shape.channels; ++c)
                            positions.push_back({n, h, w, c});
                    }
                }
            }
            return positions;
        }

        /** @brief Checks that pixel_coord returns, for flat index i, the i-th listed position. */
        static void expect_coords_follow_the_listing(const ShapeRow& shape)
        {
            const std::vector<Kernels::Indexing::PixelCoord> listed = listed_positions(shape);
            for (std::size_t i = 0; i < listed.size(); ++i)
            {
                const Kernels::Indexing::PixelCoord at = Kernels::Indexing::pixel_coord(
                    static_cast<int64_t>(i), shape.height, shape.width, shape.channels);
                EXPECT_EQ(at.n, listed[i].n) << "flat " << i;
                EXPECT_EQ(at.h, listed[i].h) << "flat " << i;
                EXPECT_EQ(at.w, listed[i].w) << "flat " << i;
                EXPECT_EQ(at.c, listed[i].c) << "flat " << i;
            }
        }

        /** @brief Checks that flat_pixel returns the listing index of each listed position. */
        static void expect_flat_pixel_is_the_listing_index(const ShapeRow& shape)
        {
            const std::vector<Kernels::Indexing::PixelCoord> listed = listed_positions(shape);
            for (std::size_t i = 0; i < listed.size(); ++i)
            {
                EXPECT_EQ(Kernels::Indexing::flat_pixel(listed[i], shape.height, shape.width,
                                                        shape.channels),
                          static_cast<int64_t>(i));
            }
        }

        class Small_PixelIndexRows : public ::testing::TestWithParam<ShapeRow>
        {
        };

        TEST_P(Small_PixelIndexRows, PixelCoordFollowsTheNestedLoopOrder)
        {
            expect_coords_follow_the_listing(GetParam());
        }

        TEST_P(Small_PixelIndexRows, FlatPixelIsTheListingIndex)
        {
            expect_flat_pixel_is_the_listing_index(GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_PixelIndexRows,
                                 ::testing::Values(ShapeRow{0, 3, 3, 3}, ShapeRow{2, 0, 3, 3},
                                                   ShapeRow{1, 1, 1, 1}, ShapeRow{2, 3, 5, 9},
                                                   ShapeRow{3, 5, 9, 1}, ShapeRow{1, 9, 2, 3},
                                                   ShapeRow{2, 1, 1, 5}, ShapeRow{5, 2, 3, 2}));

        TEST(Small_PixelIndex, StridesOffsetIsTheWeightedSumOfTheFourCoordinates)
        {
            // 1 * 100 + 2 * 10 + 3 * 3 + 4 * 2 = 137.
            const Kernels::Indexing::Strides strides{100, 10, 3, 2};

            EXPECT_EQ(strides.offset(1, 2, 3, 4), 137);
        }

        TEST(Small_PixelIndex, InputStridesTakeTheInputFieldsOfTheSpan)
        {
            Spatial::WindowSpan span;
            span.in_batch_stride = 11;
            span.in_row_stride = 12;
            span.in_col_stride = 13;
            span.in_channel_stride = 14;
            span.out_batch_stride = 21;
            span.out_row_stride = 22;
            span.out_col_stride = 23;
            span.out_channel_stride = 24;

            const Kernels::Indexing::Strides strides = Kernels::Indexing::input_strides(span);

            EXPECT_EQ(strides.batch, 11);
            EXPECT_EQ(strides.row, 12);
            EXPECT_EQ(strides.col, 13);
            EXPECT_EQ(strides.channel, 14);
        }

        TEST(Small_PixelIndex, OutputStridesTakeTheOutputFieldsOfTheSpan)
        {
            Spatial::WindowSpan span;
            span.in_batch_stride = 11;
            span.in_row_stride = 12;
            span.in_col_stride = 13;
            span.in_channel_stride = 14;
            span.out_batch_stride = 21;
            span.out_row_stride = 22;
            span.out_col_stride = 23;
            span.out_channel_stride = 24;

            const Kernels::Indexing::Strides strides = Kernels::Indexing::output_strides(span);

            EXPECT_EQ(strides.batch, 21);
            EXPECT_EQ(strides.row, 22);
            EXPECT_EQ(strides.col, 23);
            EXPECT_EQ(strides.channel, 24);
        }

        TEST(Small_PixelIndex, BatchChannelOffsetIsTheWeightedSumOfBothCoordinates)
        {
            // 2 * 7 + 3 * 5 = 29.
            EXPECT_EQ(Kernels::Indexing::batch_channel_offset(2, 3, 7, 5), 29);
        }
    }
}
