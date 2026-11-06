/**************************************************************************/
/* test_fixed_order_reduce.cpp                                            */
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
 * @file test_fixed_order_reduce.cpp
 * @brief Tests the level plan and the tile folds of the fixed-order reduction, without a device.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <SushiBLAS/graph_primitives/fixed_order_reduce.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using GraphPrimitives::fold_indexed_tile;
            using GraphPrimitives::fold_tile;
            using GraphPrimitives::Detail::plan_reduction;
            using GraphPrimitives::Detail::reduce_scratch_count;

            /// @brief Holds the tile size every expected number below derives from.
            constexpr std::size_t K_TILE = GraphPrimitives::K_REDUCE_TILE;

            /// @brief Holds a capacity of three levels.
            constexpr std::size_t K_THREE_LEVEL_CAPACITY = 1000000;

            /// @brief Names a tile and the sum of ones it must fold.
            struct TileRow
            {
                /// @brief Holds the tile index.
                std::size_t tile;
                /// @brief Holds the number of ones the tile covers.
                std::uint64_t folded;
            };

            /** @brief Appends one decimal digit per element, so a fold shows its order. */
            struct AppendDigit
            {
                /** @brief Returns the accumulator with the digit of the element appended. */
                std::uint64_t operator()(std::uint64_t accumulator, std::uint64_t element) const
                {
                    return accumulator * 10 + element;
                }
            };
        } // namespace

        /** @brief Tests that a count of zero plans no level. */
        TEST(Small_GraphPrimitivesFixedOrderLevels, AnEmptyPlanHasNoLevel)
        {
            EXPECT_TRUE(plan_reduction(0).empty());
        }

        /** @brief Tests that one element plans one level that needs no scratch. */
        TEST(Small_GraphPrimitivesFixedOrderLevels, ASingleElementPlansOneLevelWithoutScratch)
        {
            const auto one = plan_reduction(1);

            ASSERT_EQ(one.size(), 1u);
            EXPECT_EQ(one[0].in_count, 1u);
            EXPECT_EQ(one[0].out_count, 1u);
            EXPECT_EQ(reduce_scratch_count(one), 0u) << "a single level writes the output";
        }

        /** @brief Tests that exactly one tile still needs only one level. */
        TEST(Small_GraphPrimitivesFixedOrderLevels, ExactlyOneTilePlansOneLevel)
        {
            EXPECT_EQ(plan_reduction(K_TILE).size(), 1u);
        }

        /** @brief Tests that a three-level count shrinks by one tile per level. */
        TEST(Small_GraphPrimitivesFixedOrderLevels, AThreeLevelCountShrinksByOneTilePerLevel)
        {
            const auto three = plan_reduction(K_THREE_LEVEL_CAPACITY);

            ASSERT_EQ(three.size(), 3u);
            EXPECT_EQ(three[0].in_count, K_THREE_LEVEL_CAPACITY);
            EXPECT_EQ(three[0].out_count, (K_THREE_LEVEL_CAPACITY + K_TILE - 1) / K_TILE);
            EXPECT_EQ(three[1].in_count, three[0].out_count);
            EXPECT_EQ(three[2].out_count, 1u);
        }

        /** @brief Tests that a level reads what the level before wrote, and the windows differ. */
        TEST(Small_GraphPrimitivesFixedOrderLevels, EachLevelReadsWhatThePreviousLevelWrote)
        {
            const auto three = plan_reduction(K_THREE_LEVEL_CAPACITY);

            ASSERT_EQ(three.size(), 3u);
            EXPECT_EQ(three[1].in_offset, three[0].out_offset);
            EXPECT_EQ(three[2].in_offset, three[1].out_offset);
            EXPECT_NE(three[0].out_offset, three[1].out_offset);
        }

        /** @brief Tests that the scratch count is the sum of the widths the levels write. */
        TEST(Small_GraphPrimitivesFixedOrderLevels, ScratchHoldsEveryLevelButTheLast)
        {
            const auto three = plan_reduction(K_THREE_LEVEL_CAPACITY);

            ASSERT_EQ(three.size(), 3u);
            EXPECT_EQ(reduce_scratch_count(three), three[0].out_count + three[1].out_count);
        }

        /** @brief Tests that a tile folds left to right from its first element. */
        TEST(Small_GraphPrimitivesFixedOrderLevels, FoldTileCombinesLeftToRight)
        {
            const std::vector<std::uint64_t> digits = {7, 1, 2, 3, 4, 9};

            EXPECT_EQ(fold_tile(AppendDigit{}, digits.data(), 1, 5), 1234u);
        }

        /** @brief Tests that a tile of one element is that element. */
        TEST(Small_GraphPrimitivesFixedOrderLevels, ATileOfOneElementIsThatElement)
        {
            const std::vector<std::uint64_t> digits = {7, 1};

            EXPECT_EQ(fold_tile(AppendDigit{}, digits.data(), 0, 1), 7u);
        }

        /// @brief Runs one tile index against a count of two tiles and three elements.
        class Small_GraphPrimitivesFixedOrderTile : public ::testing::TestWithParam<TileRow>
        {
        };

        /** @brief Tests that tile t covers K_TILE values and the last tile stops at the count. */
        TEST_P(Small_GraphPrimitivesFixedOrderTile, IndexedTileStopsAtTheLiveCount)
        {
            const std::size_t count = 2 * K_TILE + 3;
            const std::vector<std::uint64_t> ones(count + K_TILE, 1);
            const auto add = [](std::uint64_t a, std::uint64_t b) { return a + b; };

            EXPECT_EQ(fold_indexed_tile(add, ones.data(), GetParam().tile, count),
                      GetParam().folded);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesFixedOrderTile,
                                 ::testing::Values(TileRow{0, K_TILE}, TileRow{1, K_TILE},
                                                   TileRow{2, 3u}));

    } // namespace Test
} // namespace SushiBLAS
