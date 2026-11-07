/**************************************************************************/
/* test_scan_passes.cpp                                                   */
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
 * @file test_scan_passes.cpp
 * @brief Tests the arithmetic of the passes a prefix scan records, without a device.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <SushiBLAS/graph_primitives/fixed_order_fold.hpp>
#include <SushiBLAS/graph_primitives/scan_passes.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            namespace Passes = GraphPrimitives::Detail;
            using Words = std::vector<std::uint32_t>;

            /// @brief Holds the tile size a level is cut into.
            constexpr std::size_t K_TILE = GraphPrimitives::K_REDUCE_TILE;

            /// @brief Holds a value no pass writes.
            constexpr std::uint32_t K_CANARY = 0xDEADBEEFu;

            /// @brief Names counts and the exclusive prefixes and sum the top pass must give.
            struct TopPassRow
            {
                /// @brief Names the row in the test report.
                const char* name;
                /// @brief Holds the counts the pass scans.
                Words counts;
                /// @brief Holds the prefix of each count.
                Words prefixes;
                /// @brief Holds the sum of every count, modulo 2^32.
                std::uint32_t sum;
            };

            /// @brief Names a tile, its base and the elements of the level it must write.
            struct TileRow
            {
                /// @brief Names the row in the test report.
                const char* name;
                /// @brief Holds the tile index.
                std::size_t tile;
                /// @brief Holds the exclusive prefix of the tile's first value.
                std::uint32_t base;
                /// @brief Holds the first element the tile writes.
                std::size_t first;
                /// @brief Holds how many elements the tile writes.
                std::size_t written;
            };

            /// @brief Names a live count the closing pass copies the total to.
            struct CloseRow
            {
                /// @brief Names the row in the test report.
                const char* name;
                /// @brief Holds the live count.
                std::size_t live_n;
            };

            /** @brief Names a test case after its row. */
            template <typename Row>
            std::string row_name(const ::testing::TestParamInfo<Row>& info)
            {
                return info.param.name;
            }

            /** @brief Returns count copies of value. */
            Words filled(std::size_t count, std::uint32_t value)
            {
                return Words(count, value);
            }

            /** @brief Returns the expected level of a tile scan over all-one counts. */
            Words expected_tile_level(const TileRow& row, std::size_t size)
            {
                Words level(size, K_CANARY);
                for (std::size_t i = 0; i < row.written; ++i)
                    level[row.first + i] = row.base + static_cast<std::uint32_t>(i);
                return level;
            }
        } // namespace

        /** @brief Tests that an empty scan zeroes the first offset and the total only. */
        TEST(Small_GraphPrimitivesScanPasses, EmptyScanZeroesTheFirstOffsetAndTheTotal)
        {
            Words offsets = filled(2, K_CANARY);
            Words total = filled(2, K_CANARY);

            Passes::publish_empty_scan(offsets.data(), total.data());

            EXPECT_EQ(offsets, (Words{0u, K_CANARY}));
            EXPECT_EQ(total, (Words{0u, K_CANARY}));
        }

        /// @brief Runs the top pass over one row of counts.
        class Small_GraphPrimitivesScanTopPass : public ::testing::TestWithParam<TopPassRow>
        {
        };

        /** @brief Tests that the top pass writes each count's exclusive prefix and the sum. */
        TEST_P(Small_GraphPrimitivesScanTopPass, TopPassWritesTheExclusivePrefixesAndTheSum)
        {
            const TopPassRow& row = GetParam();
            Words prefixes = filled(row.counts.size(), K_CANARY);
            Words total = filled(1, K_CANARY);

            Passes::scan_top_pass(row.counts.data(), prefixes.data(), row.counts.size(),
                                  total.data(), nullptr);

            EXPECT_EQ(prefixes, row.prefixes);
            EXPECT_EQ(total[0], row.sum);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_GraphPrimitivesScanTopPass,
            ::testing::Values(TopPassRow{"NoCount", {}, {}, 0u},
                              TopPassRow{"OneCount", {7u}, {0u}, 7u},
                              TopPassRow{"FourCounts", {1u, 2u, 3u, 4u}, {0u, 1u, 3u, 6u}, 10u},
                              TopPassRow{"SumWrapsModulo2To32",
                                         {0xFFFFFFFFu, 2u},
                                         {0u, 0xFFFFFFFFu},
                                         1u}),
            row_name<TopPassRow>);

        /** @brief Tests that the first level's top pass also writes the sum at offsets[n]. */
        TEST(Small_GraphPrimitivesScanTopPass, FirstLevelPublishesTheSumAfterTheLastOffset)
        {
            const Words counts{1u, 2u, 3u};
            Words offsets = filled(5, K_CANARY);
            Words total = filled(1, K_CANARY);

            Passes::scan_top_pass(counts.data(), offsets.data(), counts.size(), total.data(),
                                  offsets.data());

            EXPECT_EQ(offsets, (Words{0u, 1u, 3u, 6u, K_CANARY}));
        }

        /** @brief Tests that a top pass above level 0 writes no sentinel. */
        TEST(Small_GraphPrimitivesScanTopPass, UpperLevelWritesNoSentinel)
        {
            const Words counts{1u, 2u, 3u};
            Words prefixes = filled(4, K_CANARY);
            Words total = filled(1, K_CANARY);

            Passes::scan_top_pass(counts.data(), prefixes.data(), counts.size(), total.data(),
                                  nullptr);

            EXPECT_EQ(prefixes, (Words{0u, 1u, 3u, K_CANARY}));
        }

        /// @brief Runs one tile of a level that holds one tile and three more values.
        class Small_GraphPrimitivesScanTile : public ::testing::TestWithParam<TileRow>
        {
        };

        /** @brief Tests that a tile counts up from its base and writes nothing outside itself. */
        TEST_P(Small_GraphPrimitivesScanTile, TileScanStartsFromItsBaseAndStaysInsideItself)
        {
            const TileRow& row = GetParam();
            const std::size_t size = K_TILE + 3;
            const Words ones = filled(size, 1u);
            Words level = filled(size, K_CANARY);

            Passes::scan_tile(ones.data(), level.data(), row.tile, size, row.base);

            EXPECT_EQ(level, expected_tile_level(row, size));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesScanTile,
                                 ::testing::Values(TileRow{"FirstTileFromZero", 0, 0u, 0, K_TILE},
                                                   TileRow{"FirstTileFromABase", 0, 10u, 0, K_TILE},
                                                   TileRow{"ShortLastTile", 1, 500u, K_TILE, 3}),
                                 row_name<TileRow>);

        /// @brief Runs the closing pass at one live count.
        class Small_GraphPrimitivesScanClose : public ::testing::TestWithParam<CloseRow>
        {
        };

        /** @brief Tests that the closing pass copies the total to offsets[live n] only. */
        TEST_P(Small_GraphPrimitivesScanClose, ClosePassCopiesTheTotalToTheLiveEnd)
        {
            const CloseRow& row = GetParam();
            Words offsets = filled(5, K_CANARY);
            const Words total{42u};

            Passes::close_scan(offsets.data(), total.data(), row.live_n);

            Words expected = filled(5, K_CANARY);
            expected[row.live_n] = 42u;
            EXPECT_EQ(offsets, expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesScanClose,
                                 ::testing::Values(CloseRow{"Empty", 0}, CloseRow{"ThreeLive", 3}),
                                 row_name<CloseRow>);

    } // namespace Test
} // namespace SushiBLAS
