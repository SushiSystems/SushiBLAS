/**************************************************************************/
/* test_upsample_index.cpp                                                */
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
 * @file test_upsample_index.cpp
 * @brief Tests the index math of nearest-neighbour upsampling against hand values and the block it repeats.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/spatial_kernels/upsample_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Indexing = Kernels::Indexing;

        /** @brief Holds one upsample_source call. */
        struct SourceRow
        {
            int64_t out_pos;
            int64_t scale;
            int64_t expected;
        };

        class Small_UpsampleIndexSource : public ::testing::TestWithParam<SourceRow>
        {
        };

        TEST_P(Small_UpsampleIndexSource, UpsampleSourceIsTheFlooredQuotient)
        {
            const SourceRow row = GetParam();

            EXPECT_EQ(Indexing::upsample_source(row.out_pos, row.scale), row.expected);
        }

        // out_pos / scale, floored: 0 / 2, 1 / 2, 2 / 2, 5 / 3, 8 / 3, 4 / 1, 0 / 3.
        INSTANTIATE_TEST_SUITE_P(, Small_UpsampleIndexSource,
                                 ::testing::Values(SourceRow{0, 2, 0}, SourceRow{1, 2, 0},
                                                   SourceRow{2, 2, 1}, SourceRow{5, 3, 1},
                                                   SourceRow{8, 3, 2}, SourceRow{4, 1, 4},
                                                   SourceRow{0, 3, 0}));

        /** @brief Holds one upsample_block call. */
        struct BlockRow
        {
            int64_t in_pos;
            int64_t scale;
            int64_t step;
            int64_t expected;
        };

        class Small_UpsampleIndexBlock : public ::testing::TestWithParam<BlockRow>
        {
        };

        TEST_P(Small_UpsampleIndexBlock, UpsampleBlockIsPositionTimesScalePlusStep)
        {
            const BlockRow row = GetParam();

            EXPECT_EQ(Indexing::upsample_block(row.in_pos, row.scale, row.step), row.expected);
        }

        // in_pos * scale + step: 2 * 3 + 1, 0 * 2 + 1, 4 * 1 + 0, 5 * 3 + 2, 0 * 1 + 0.
        INSTANTIATE_TEST_SUITE_P(, Small_UpsampleIndexBlock,
                                 ::testing::Values(BlockRow{2, 3, 1, 7}, BlockRow{0, 2, 1, 1},
                                                   BlockRow{4, 1, 0, 4}, BlockRow{5, 3, 2, 17},
                                                   BlockRow{0, 1, 0, 0}));

        namespace
        {
            /** @brief Holds the input extent and the scale of one axis. */
            struct AxisRow
            {
                int64_t in_extent;
                int64_t scale;
            };
        } // namespace

        /** @brief Checks that every output position of each block maps back to the block's input position. */
        static void expect_blocks_map_back(const AxisRow& axis)
        {
            for (int64_t in_pos = 0; in_pos < axis.in_extent; ++in_pos)
            {
                for (int64_t step = 0; step < axis.scale; ++step)
                {
                    EXPECT_EQ(Indexing::upsample_source(
                                  Indexing::upsample_block(in_pos, axis.scale, step), axis.scale),
                              in_pos)
                        << "in_pos " << in_pos << " step " << step;
                }
            }
        }

        class Small_UpsampleIndexAxes : public ::testing::TestWithParam<AxisRow>
        {
        };

        TEST_P(Small_UpsampleIndexAxes, EveryPositionOfABlockMapsBackToItsSource)
        {
            expect_blocks_map_back(GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_UpsampleIndexAxes,
                                 ::testing::Values(AxisRow{0, 2}, AxisRow{1, 1}, AxisRow{1, 3},
                                                   AxisRow{2, 2}, AxisRow{3, 3}, AxisRow{5, 2},
                                                   AxisRow{9, 3}, AxisRow{9, 1}));
    }
}
