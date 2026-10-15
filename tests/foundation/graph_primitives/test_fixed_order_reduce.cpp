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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
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

        /** @brief Tests that the plan depends on the count alone and its levels chain. */
        TEST(Medium_GraphPrimitivesFixedOrderLevels, PlanLayoutIsPureAndDisjoint)
        {
            EXPECT_TRUE(plan_reduction(0).empty());

            const auto one = plan_reduction(1);
            ASSERT_EQ(one.size(), 1u);
            EXPECT_EQ(one[0].in_count, 1u);
            EXPECT_EQ(one[0].out_count, 1u);
            EXPECT_EQ(reduce_scratch_count(one), 0u) << "a single level writes the output";

            const auto exact = plan_reduction(K_TILE);
            ASSERT_EQ(exact.size(), 1u) << "exactly one tile still needs only one level";

            const auto three = plan_reduction(K_THREE_LEVEL_CAPACITY);
            ASSERT_EQ(three.size(), 3u);
            EXPECT_EQ(three[0].in_count, K_THREE_LEVEL_CAPACITY);
            EXPECT_EQ(three[0].out_count, (K_THREE_LEVEL_CAPACITY + K_TILE - 1) / K_TILE);
            EXPECT_EQ(three[1].in_count, three[0].out_count);
            EXPECT_EQ(three[2].out_count, 1u);

            // Level i reads what level i - 1 wrote, and the two windows do not alias.
            for (std::size_t i = 1; i < three.size(); ++i)
                EXPECT_EQ(three[i].in_offset, three[i - 1].out_offset) << "at level " << i;
            EXPECT_NE(three[0].out_offset, three[1].out_offset);

            EXPECT_EQ(reduce_scratch_count(three), three[0].out_count + three[1].out_count);
        }

        /** @brief Tests that a tile folds left to right from its first element. */
        TEST(Medium_GraphPrimitivesFixedOrderLevels, FoldTileCombinesLeftToRight)
        {
            const std::vector<std::uint64_t> digits = {7, 1, 2, 3, 4, 9};

            EXPECT_EQ(fold_tile(AppendDigit{}, digits.data(), 1, 5), 1234u);
            EXPECT_EQ(fold_tile(AppendDigit{}, digits.data(), 0, 1), 7u)
                << "a tile of one element is that element";
        }

        /** @brief Tests that tile t covers K_TILE values and the last tile stops at the count. */
        TEST(Medium_GraphPrimitivesFixedOrderLevels, IndexedTileStopsAtTheLiveCount)
        {
            const std::size_t count = 2 * K_TILE + 3;
            std::vector<std::uint64_t> ones(count + K_TILE, 1);

            const auto add = [](std::uint64_t a, std::uint64_t b) { return a + b; };
            EXPECT_EQ(fold_indexed_tile(add, ones.data(), 0, count), K_TILE);
            EXPECT_EQ(fold_indexed_tile(add, ones.data(), 1, count), K_TILE);
            EXPECT_EQ(fold_indexed_tile(add, ones.data(), 2, count), 3u)
                << "the last tile must not read past the count";
        }
    } // namespace Test
} // namespace SushiBLAS
