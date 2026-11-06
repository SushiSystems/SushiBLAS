/**************************************************************************/
/* test_fixed_order_scan.cpp                                              */
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
 * @file test_fixed_order_scan.cpp
 * @brief Tests the tiled level plan and the live level widths of the fixed-order scan.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/graph_primitives/fixed_order_reduce.hpp>
#include <SushiBLAS/graph_primitives/fixed_order_scan.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using GraphPrimitives::Detail::fill_live_level_words;
            using GraphPrimitives::Detail::live_tile_levels;
            using GraphPrimitives::Detail::next_live_tile_count;
            using GraphPrimitives::Detail::plan_reduction;
            using GraphPrimitives::Detail::plan_tiled_levels;

            /// @brief Holds the tile size every expected number below derives from.
            constexpr std::size_t K_TILE = GraphPrimitives::K_REDUCE_TILE;

            /// @brief Holds a capacity of three levels.
            constexpr std::size_t K_THREE_LEVEL_CAPACITY = 1000000;

            /// @brief Names a capacity and the number of levels that fold it.
            struct LevelRow
            {
                /// @brief Holds the capacity in elements.
                std::size_t capacity;
                /// @brief Holds the number of levels the plan has.
                std::size_t levels;
            };

            /** @brief Returns whether each live level has the width the fixed plan gives it. */
            template <typename Live, typename Fixed>
            bool live_matches_fixed(const Live& live, const Fixed& fixed)
            {
                for (std::size_t i = 0; i < fixed.size(); ++i)
                {
                    if (live[i].in_count != fixed[i].in_count ||
                        live[i].out_count != fixed[i].out_count)
                        return false;
                }
                return true;
            }

            /** @brief Returns whether every level from first on takes and gives one value. */
            template <typename Live>
            bool passes_one_value_from(const Live& live, std::size_t first)
            {
                for (std::size_t i = first; i < live.size(); ++i)
                {
                    if (live[i].in_count != 1u || live[i].out_count != 1u)
                        return false;
                }
                return true;
            }

            /** @brief Returns whether no level takes or gives a value. */
            template <typename Live>
            bool all_levels_empty(const Live& live)
            {
                for (const auto& level : live)
                {
                    if (level.in_count != 0u || level.out_count != 0u)
                        return false;
                }
                return true;
            }

            /** @brief Returns the input width of each level. */
            template <typename Live>
            std::vector<std::size_t> in_counts_of(const Live& live)
            {
                std::vector<std::size_t> counts;
                for (const auto& level : live)
                    counts.push_back(static_cast<std::size_t>(level.in_count));
                return counts;
            }

            /** @brief Returns the output width of each level. */
            template <typename Live>
            std::vector<std::size_t> out_counts_of(const Live& live)
            {
                std::vector<std::size_t> counts;
                for (const auto& level : live)
                    counts.push_back(static_cast<std::size_t>(level.out_count));
                return counts;
            }

            /** @brief Returns count words of the flat array from begin on. */
            std::vector<std::size_t> words_from(const std::vector<std::uint32_t>& words,
                                                std::size_t begin, std::size_t count)
            {
                return std::vector<std::size_t>(words.begin() + begin,
                                                words.begin() + begin + count);
            }
        } // namespace

        /// @brief Runs one capacity against the tiled level plan.
        class Small_GraphPrimitivesFixedOrderLevelCount : public ::testing::TestWithParam<LevelRow>
        {
        };

        /** @brief Tests the level count of a capacity at the tile-size boundaries. */
        TEST_P(Small_GraphPrimitivesFixedOrderLevelCount, LevelCountsAtTileBoundaries)
        {
            EXPECT_EQ(plan_tiled_levels(GetParam().capacity).size(), GetParam().levels);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_GraphPrimitivesFixedOrderLevelCount,
            ::testing::Values(LevelRow{0, 0u}, LevelRow{1, 1u}, LevelRow{K_TILE - 1, 1u},
                              LevelRow{K_TILE, 1u}, LevelRow{K_TILE + 1, 2u},
                              LevelRow{K_TILE * K_TILE, 2u}, LevelRow{K_TILE * K_TILE + 1, 3u},
                              LevelRow{K_THREE_LEVEL_CAPACITY, 3u}));

        /// @brief Runs one live count against the plan laid out for the capacity.
        class Small_GraphPrimitivesFixedOrderLiveCount
            : public ::testing::TestWithParam<std::size_t>
        {
        };

        /** @brief Tests that a live count folds the levels plan_reduction lays out for it. */
        TEST_P(Small_GraphPrimitivesFixedOrderLiveCount, LiveCountBelowCapacityMatchesThatCount)
        {
            const std::size_t n = GetParam();
            const auto capacity_levels = plan_tiled_levels(K_THREE_LEVEL_CAPACITY);

            const auto live = live_tile_levels(capacity_levels, n);
            const auto fixed = plan_reduction(n);

            ASSERT_EQ(live.size(), capacity_levels.size());
            ASSERT_LE(fixed.size(), live.size());
            EXPECT_TRUE(live_matches_fixed(live, fixed));
        }

        /** @brief Tests that a level past what the live count needs passes one value through. */
        TEST_P(Small_GraphPrimitivesFixedOrderLiveCount, LevelsPastTheLiveCountPassOneValueThrough)
        {
            const std::size_t n = GetParam();
            const auto capacity_levels = plan_tiled_levels(K_THREE_LEVEL_CAPACITY);

            const auto live = live_tile_levels(capacity_levels, n);
            const auto fixed = plan_reduction(n);

            ASSERT_EQ(live.size(), capacity_levels.size());
            EXPECT_TRUE(passes_one_value_from(live, fixed.size()));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesFixedOrderLiveCount,
                                 ::testing::Values(1, K_TILE - 1, K_TILE, K_TILE + 1,
                                                   K_TILE * K_TILE, K_TILE * K_TILE + 1));

        /** @brief Tests that a live count of zero folds no level anywhere in the tree. */
        TEST(Small_GraphPrimitivesFixedOrderLevels, LiveCountZeroFoldsNothing)
        {
            const auto live = live_tile_levels(plan_tiled_levels(K_THREE_LEVEL_CAPACITY), 0);

            ASSERT_EQ(live.size(), 3u);
            EXPECT_TRUE(all_levels_empty(live));
            EXPECT_EQ(next_live_tile_count(0), 0u);
        }

        /// @brief Runs one live count against the flat level words.
        class Small_GraphPrimitivesFixedOrderLevelWords
            : public ::testing::TestWithParam<std::size_t>
        {
        };

        /** @brief Tests that the flat words a kernel reads equal the per-level widths. */
        TEST_P(Small_GraphPrimitivesFixedOrderLevelWords, LiveLevelWordsMatchTheLiveLevels)
        {
            const std::size_t n = GetParam();
            const auto capacity_levels = plan_tiled_levels(K_THREE_LEVEL_CAPACITY);
            const std::size_t level_count = capacity_levels.size();
            const auto live = live_tile_levels(capacity_levels, n);
            std::vector<std::uint32_t> words(2 * level_count, 0xFFFFFFFFu);

            fill_live_level_words(level_count, n, words.data());

            EXPECT_EQ(words_from(words, 0, level_count), in_counts_of(live));
            EXPECT_EQ(words_from(words, level_count, level_count), out_counts_of(live));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesFixedOrderLevelWords,
                                 ::testing::Values(0, 1, K_TILE + 1, K_TILE * K_TILE + 1));

    } // namespace Test
} // namespace SushiBLAS
