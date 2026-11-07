/**************************************************************************/
/* test_sort_resolution.cpp                                               */
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
 * @file test_sort_resolution.cpp
 * @brief Tests what a counting sort decides from its live sizes, without a device.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <SushiBLAS/graph_primitives/radix_plan.hpp>
#include <SushiBLAS/graph_primitives/sort_resolution.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            namespace Sort = GraphPrimitives::Detail;
            using Words = std::vector<std::uint32_t>;

            /// @brief Holds a value no decision writes.
            constexpr std::uint32_t K_CANARY = 0xDEADBEEFu;

            /// @brief Holds the digits the plans below were made for.
            constexpr std::size_t K_DIGITS = 2;

            /// @brief Names the live sizes of a run and what the state and the plan must say.
            struct StateRow
            {
                /// @brief Names the row in the test report.
                const char* name;
                /// @brief Holds the live element count.
                std::size_t live_n;
                /// @brief Holds the live key range.
                std::size_t live_key_range;
                /// @brief Holds the live tile count of the elements.
                std::uint32_t tiles;
                /// @brief Holds the tiles the second digit runs: zero when the range skips it.
                std::uint32_t second_digit_tiles;
            };

            /// @brief Names a bucket of a sort and the table size it is cleared in.
            struct BucketRow
            {
                /// @brief Names the row in the test report.
                const char* name;
                /// @brief Holds the bucket.
                std::size_t k;
                /// @brief Holds whether counts[k] is cleared too.
                bool counts_cleared;
            };

            /// @brief Names a live count, a bucket and the place the bucket starts at.
            struct StartRow
            {
                /// @brief Names the row in the test report.
                const char* name;
                /// @brief Holds the live element count.
                std::size_t live_n;
                /// @brief Holds the bucket.
                std::size_t k;
                /// @brief Holds the number of live keys below the bucket.
                std::uint32_t start;
            };

            /** @brief Names a test case after its row. */
            template <typename Row>
            std::string row_name(const ::testing::TestParamInfo<Row>& info)
            {
                return info.param.name;
            }

            /** @brief Returns the plan word of a digit. */
            std::uint32_t plan_word(const Words& plan, std::size_t digit, std::size_t word)
            {
                return plan[digit * Sort::K_RADIX_PLAN_WORDS + word];
            }
        } // namespace

        /// @brief Publishes the state of one run.
        class Small_GraphPrimitivesSortState : public ::testing::TestWithParam<StateRow>
        {
        };

        /** @brief Tests that the live count, its tiles and the total reach the state. */
        TEST_P(Small_GraphPrimitivesSortState, StateHoldsTheLiveCountItsTilesAndTheTotal)
        {
            const StateRow& row = GetParam();
            Words n_state(2, K_CANARY);
            Words total(1, K_CANARY);
            Words plan(K_DIGITS * Sort::K_RADIX_PLAN_WORDS, K_CANARY);

            Sort::publish_sort_state(K_DIGITS, row.live_n, row.live_key_range, n_state.data(),
                                     total.data(), plan.data());

            EXPECT_EQ(n_state, (Words{static_cast<std::uint32_t>(row.live_n), row.tiles}));
            EXPECT_EQ(total[0], static_cast<std::uint32_t>(row.live_n));
        }

        /** @brief Tests that the plan the state fills skips the digits the key range spares. */
        TEST_P(Small_GraphPrimitivesSortState, PlanRunsOnlyTheDigitsTheLiveKeyRangeNeeds)
        {
            const StateRow& row = GetParam();
            Words n_state(2, K_CANARY);
            Words total(1, K_CANARY);
            Words plan(K_DIGITS * Sort::K_RADIX_PLAN_WORDS, K_CANARY);

            Sort::publish_sort_state(K_DIGITS, row.live_n, row.live_key_range, n_state.data(),
                                     total.data(), plan.data());

            EXPECT_EQ(plan_word(plan, 0, Sort::K_RADIX_PLAN_TILES), row.tiles);
            EXPECT_EQ(plan_word(plan, 1, Sort::K_RADIX_PLAN_TILES), row.second_digit_tiles);
            EXPECT_EQ(plan_word(plan, 1, Sort::K_RADIX_PLAN_LIVE_N),
                      static_cast<std::uint32_t>(row.live_n));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesSortState,
                                 ::testing::Values(StateRow{"NoElement", 0, 300, 0u, 0u},
                                                   StateRow{"OneByteRange", 1000, 16, 4u, 0u},
                                                   StateRow{"TwoByteRange", 1000, 300, 4u, 4u},
                                                   StateRow{"SingleTile", 5, 300, 1u, 1u}),
                                 row_name<StateRow>);

        /** @brief Tests that a sort with no plan still publishes its count, tiles and total. */
        TEST(Small_GraphPrimitivesSortState, StateIsPublishedWithoutAPlan)
        {
            Words n_state(2, K_CANARY);
            Words total(1, K_CANARY);

            Sort::publish_sort_state(K_DIGITS, 300, 16, n_state.data(), total.data(), nullptr);

            EXPECT_EQ(n_state, (Words{300u, 2u}));
            EXPECT_EQ(total[0], 300u);
        }

        /// @brief Clears one bucket of a sort planned for four buckets.
        class Small_GraphPrimitivesSortEmptyBucket : public ::testing::TestWithParam<BucketRow>
        {
        };

        /** @brief Tests that an empty sort zeroes the offset of a bucket and its count. */
        TEST_P(Small_GraphPrimitivesSortEmptyBucket, EmptyBucketIsZeroedInOffsetsAndCounts)
        {
            constexpr std::size_t K_BUCKETS = 4;
            const BucketRow& row = GetParam();
            Words offsets(K_BUCKETS + 1, K_CANARY);
            Words counts(K_BUCKETS + 1, K_CANARY);

            Sort::clear_empty_bucket(row.k, K_BUCKETS, offsets.data(), counts.data());

            Words expected_offsets(K_BUCKETS + 1, K_CANARY);
            expected_offsets[row.k] = 0u;
            Words expected_counts(K_BUCKETS + 1, K_CANARY);
            expected_counts[row.k] = row.counts_cleared ? 0u : K_CANARY;
            EXPECT_EQ(offsets, expected_offsets);
            EXPECT_EQ(counts, expected_counts);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesSortEmptyBucket,
                                 ::testing::Values(BucketRow{"FirstBucket", 0, true},
                                                   BucketRow{"LastBucket", 3, true},
                                                   BucketRow{"ClosingOffsetHasNoCount", 4, false}),
                                 row_name<BucketRow>);

        /** @brief Tests that an empty sort without a counts table clears the offset alone. */
        TEST(Small_GraphPrimitivesSortEmptyBucket, NoCountsTableIsLeftAlone)
        {
            Words offsets(5, K_CANARY);

            Sort::clear_empty_bucket(2, 4, offsets.data(), nullptr);

            EXPECT_EQ(offsets, (Words{K_CANARY, K_CANARY, 0u, K_CANARY, K_CANARY}));
        }

        /// @brief Looks up where one bucket starts in a sorted run.
        class Small_GraphPrimitivesSortBucketStart : public ::testing::TestWithParam<StartRow>
        {
        };

        /** @brief Tests that a bucket starts after every live key below it. */
        TEST_P(Small_GraphPrimitivesSortBucketStart, BucketStartsAfterTheLiveKeysBelowIt)
        {
            // The keys sorted by index order 1, 4, 3, 0, 2 read 1, 1, 3, 5, 5.
            const Words keys{5u, 1u, 5u, 3u, 1u};
            const Words order{1u, 4u, 3u, 0u, 2u};
            const StartRow& row = GetParam();

            EXPECT_EQ(Sort::bucket_start(keys.data(), order.data(), row.live_n, row.k),
                      row.start);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_GraphPrimitivesSortBucketStart,
            ::testing::Values(StartRow{"BelowTheSmallestKey", 5, 0, 0u},
                              StartRow{"AtTheSmallestKey", 5, 1, 0u},
                              StartRow{"BetweenKeys", 5, 2, 2u},
                              StartRow{"AtAKeyWithOneElement", 5, 3, 2u},
                              StartRow{"BetweenTwoLaterKeys", 5, 4, 3u},
                              StartRow{"AtADuplicatedKey", 5, 5, 3u},
                              StartRow{"PastTheLargestKey", 5, 6, 5u},
                              StartRow{"OnlyTheLiveKeysCount", 3, 5, 3u},
                              StartRow{"NoLiveKey", 0, 7, 0u}),
            row_name<StartRow>);

    } // namespace Test
} // namespace SushiBLAS
