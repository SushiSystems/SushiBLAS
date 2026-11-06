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

            /** @brief Returns the left fold of values[begin, end) under Mix. */
            std::uint64_t left_fold(const std::vector<std::uint64_t>& values, std::size_t begin,
                                    std::size_t end)
            {
                std::uint64_t folded = values[begin];
                for (std::size_t i = begin + 1; i < end; ++i)
                    folded = Mix{}(folded, values[i]);
                return folded;
            }

            /** @brief Returns the left fold of each tile of values, in order. */
            std::vector<std::uint64_t> tile_folds_of(const std::vector<std::uint64_t>& values)
            {
                std::vector<std::uint64_t> folds;
                for (std::size_t begin = 0; begin < values.size(); begin += K_TILE)
                {
                    const std::size_t end =
                        begin + K_TILE < values.size() ? begin + K_TILE : values.size();
                    folds.push_back(left_fold(values, begin, end));
                }
                return folds;
            }

            /** @brief Folds with the streamed form over a vector. */
            std::uint64_t streamed(const std::vector<std::uint64_t>& values)
            {
                const std::uint64_t* data = values.data();
                return fold_fixed_order_streamed(Mix{}, values.size(), K_IDENTITY,
                                                 [data](std::size_t i) { return data[i]; });
            }
        } // namespace

        /** @brief Tests that an empty fold returns the identity in both forms. */
        TEST(Small_GraphPrimitivesFixedOrderFold, AnEmptyFoldReturnsTheIdentity)
        {
            const std::vector<std::uint64_t> none;

            EXPECT_EQ(fold_fixed_order(Mix{}, none.data(), 0, K_IDENTITY), K_IDENTITY);
            EXPECT_EQ(streamed(none), K_IDENTITY);
        }

        /** @brief Tests that a fold of one element returns that element in both forms. */
        TEST(Small_GraphPrimitivesFixedOrderFold, OneElementReturnsItself)
        {
            const std::vector<std::uint64_t> one = {42u};

            EXPECT_EQ(fold_fixed_order(Mix{}, one.data(), 1, K_IDENTITY), 42u);
            EXPECT_EQ(streamed(one), 42u);
        }

        /** @brief Tests that a count within one tile folds left to right from its first element. */
        TEST(Small_GraphPrimitivesFixedOrderFold, OneTileFoldsLeftToRight)
        {
            const std::vector<std::uint64_t> values = values_of(K_TILE);
            const std::uint64_t expected = left_fold(values, 0, K_TILE);

            EXPECT_EQ(fold_fixed_order(Mix{}, values.data(), K_TILE, K_IDENTITY), expected);
            EXPECT_EQ(streamed(values), expected);
        }

        /** @brief Tests that a second tile is folded on its own before it meets the first. */
        TEST(Small_GraphPrimitivesFixedOrderFold, ASecondTileIsFoldedOnItsOwn)
        {
            const std::vector<std::uint64_t> values = values_of(K_TILE + 2);
            const std::uint64_t first_tile = left_fold(values, 0, K_TILE);
            const std::uint64_t second_tile = left_fold(values, K_TILE, K_TILE + 2);
            const std::uint64_t expected = Mix{}(first_tile, second_tile);
            ASSERT_NE(expected, left_fold(values, 0, values.size()))
                << "the operator must tell the tree from a flat fold";

            EXPECT_EQ(fold_fixed_order(Mix{}, values.data(), values.size(), K_IDENTITY), expected);
            EXPECT_EQ(streamed(values), expected);
        }

        /// @brief Runs one element count against both forms of the fold.
        class Small_GraphPrimitivesFixedOrderFoldBoundary
            : public ::testing::TestWithParam<std::size_t>
        {
        };

        /** @brief Tests that the levelled and the streamed fold agree at a level boundary. */
        TEST_P(Small_GraphPrimitivesFixedOrderFoldBoundary, BothFormsAgreeAcrossLevelBoundaries)
        {
            const std::size_t count = GetParam();
            const std::vector<std::uint64_t> values = values_of(count);

            EXPECT_EQ(fold_fixed_order(Mix{}, values.data(), count, K_IDENTITY), streamed(values));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesFixedOrderFoldBoundary,
                                 ::testing::Values(2, K_TILE - 1, K_TILE + 1, 2 * K_TILE,
                                                   K_TILE * K_TILE - 1, K_TILE * K_TILE,
                                                   K_TILE * K_TILE + 1,
                                                   K_TILE * K_TILE + K_TILE + 1,
                                                   3 * K_TILE * K_TILE + 5));

        /** @brief Tests a three-level count against the tree written out by hand. */
        TEST(Small_GraphPrimitivesFixedOrderFold, ThreeLevelsMatchTheTreeWrittenOut)
        {
            const std::size_t count = K_TILE * K_TILE + 3;
            const std::vector<std::uint64_t> values = values_of(count);
            const std::vector<std::uint64_t> level_one = tile_folds_of(values);
            ASSERT_EQ(level_one.size(), K_TILE + 1);
            const std::uint64_t expected =
                Mix{}(left_fold(level_one, 0, K_TILE), level_one[K_TILE]);

            EXPECT_EQ(fold_fixed_order(Mix{}, values.data(), count, K_IDENTITY), expected);
            EXPECT_EQ(streamed(values), expected);
        }

    } // namespace Test
} // namespace SushiBLAS
