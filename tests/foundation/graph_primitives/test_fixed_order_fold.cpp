/**************************************************************************/
/* test_fixed_order_fold.cpp                                              */
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
 * @file test_fixed_order_fold.cpp
 * @brief Tests the two host folds against the fixed tile order, without a device.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <SushiBLAS/graph_primitives/fixed_order_fold.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using GraphPrimitives::fold_fixed_order;
            using GraphPrimitives::fold_fixed_order_streamed;
            using GraphPrimitives::fold_tile;

            /// @brief Holds the tile size every expected number below derives from.
            constexpr std::size_t K_TILE = GraphPrimitives::K_REDUCE_TILE;

            /// @brief Holds the value an empty fold must return untouched.
            constexpr std::uint64_t K_IDENTITY = 0xFEEDu;

            /** @brief Mixes the element into the accumulator so a fold shows order and grouping. */
            struct Mix
            {
                /** @brief Returns a value that changes when the operands swap or regroup. */
                std::uint64_t operator()(std::uint64_t accumulator, std::uint64_t element) const
                {
                    return accumulator * 31u + element;
                }
            };

            /** @brief Returns count distinct values. */
            std::vector<std::uint64_t> values_of(std::size_t count)
            {
                std::vector<std::uint64_t> values(count);
                for (std::size_t i = 0; i < count; ++i)
                    values[i] = i * 2654435761u + 7u;
                return values;
            }

            /** @brief Folds with the streamed form over a vector. */
            std::uint64_t streamed(const std::vector<std::uint64_t>& values)
            {
                const std::uint64_t* data = values.data();
                return fold_fixed_order_streamed(Mix{}, values.size(), K_IDENTITY,
                                                 [data](std::size_t i) { return data[i]; });
            }
        } // namespace

        /** @brief Tests that an empty fold returns the identity and one element returns itself. */
        TEST(Medium_GraphPrimitivesFixedOrderFold, EmptyAndSingleCounts)
        {
            const std::vector<std::uint64_t> none;
            EXPECT_EQ(fold_fixed_order(Mix{}, none.data(), 0, K_IDENTITY), K_IDENTITY);
            EXPECT_EQ(streamed(none), K_IDENTITY);

            const std::vector<std::uint64_t> one = {42u};
            EXPECT_EQ(fold_fixed_order(Mix{}, one.data(), 1, K_IDENTITY), 42u);
            EXPECT_EQ(streamed(one), 42u);
        }

        /** @brief Tests that a count within one tile folds left to right from its first element. */
        TEST(Medium_GraphPrimitivesFixedOrderFold, OneTileFoldsLeftToRight)
        {
            const std::vector<std::uint64_t> values = values_of(K_TILE);
            std::uint64_t expected = values[0];
            for (std::size_t i = 1; i < K_TILE; ++i)
                expected = expected * 31u + values[i];

            EXPECT_EQ(fold_fixed_order(Mix{}, values.data(), K_TILE, K_IDENTITY), expected);
            EXPECT_EQ(streamed(values), expected);
        }

        /** @brief Tests that a second tile is folded on its own before it meets the first. */
        TEST(Medium_GraphPrimitivesFixedOrderFold, ASecondTileIsFoldedOnItsOwn)
        {
            const std::vector<std::uint64_t> values = values_of(K_TILE + 2);
            const std::uint64_t first_tile = fold_tile(Mix{}, values.data(), 0, K_TILE);
            const std::uint64_t second_tile = values[K_TILE] * 31u + values[K_TILE + 1];
            const std::uint64_t expected = first_tile * 31u + second_tile;

            std::uint64_t flat = values[0];
            for (std::size_t i = 1; i < values.size(); ++i)
                flat = flat * 31u + values[i];
            ASSERT_NE(expected, flat) << "the operator must tell the tree from a flat fold";

            EXPECT_EQ(fold_fixed_order(Mix{}, values.data(), values.size(), K_IDENTITY), expected);
            EXPECT_EQ(streamed(values), expected);
        }

        /** @brief Tests that the levelled and the streamed fold agree at every level boundary. */
        TEST(Medium_GraphPrimitivesFixedOrderFold, BothFormsAgreeAcrossLevelBoundaries)
        {
            const std::size_t counts[] = {2,
                                          K_TILE - 1,
                                          K_TILE + 1,
                                          2 * K_TILE,
                                          K_TILE * K_TILE - 1,
                                          K_TILE * K_TILE,
                                          K_TILE * K_TILE + 1,
                                          K_TILE * K_TILE + K_TILE + 1,
                                          3 * K_TILE * K_TILE + 5};
            for (const std::size_t count : counts)
            {
                const std::vector<std::uint64_t> values = values_of(count);
                EXPECT_EQ(fold_fixed_order(Mix{}, values.data(), count, K_IDENTITY),
                          streamed(values))
                    << "count " << count;
            }
        }

        /** @brief Tests a three-level count against the tree written out by hand. */
        TEST(Medium_GraphPrimitivesFixedOrderFold, ThreeLevelsMatchTheTreeWrittenOut)
        {
            const std::size_t count = K_TILE * K_TILE + 3;
            const std::vector<std::uint64_t> values = values_of(count);

            std::vector<std::uint64_t> level_one;
            for (std::size_t begin = 0; begin < count; begin += K_TILE)
            {
                const std::size_t end = begin + K_TILE < count ? begin + K_TILE : count;
                level_one.push_back(fold_tile(Mix{}, values.data(), begin, end));
            }
            ASSERT_EQ(level_one.size(), K_TILE + 1);
            const std::uint64_t left = fold_tile(Mix{}, level_one.data(), 0, K_TILE);
            const std::uint64_t expected = left * 31u + level_one[K_TILE];

            EXPECT_EQ(fold_fixed_order(Mix{}, values.data(), count, K_IDENTITY), expected);
            EXPECT_EQ(streamed(values), expected);
        }

    } // namespace Test
} // namespace SushiBLAS
