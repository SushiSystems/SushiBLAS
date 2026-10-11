/**************************************************************************/
/* test_counting_sort.cpp                                                 */
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
 * @file test_counting_sort.cpp
 * @brief Tests counting_sort(): stability, digit boundaries, late-bound replay and node widths.
 * @author Mustafa Garip
 * @see modules/foundation/graph_primitives/tests/README.md
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/graph/dynamic_graph.hpp>
#include <SushiRuntime/api/graph/graph.hpp>
#include <SushiRuntime/api/runtime/runtime.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>
#include <SushiRuntime/core/error.hpp>

#include <SushiBLAS/graph_primitives/counting_sort.hpp>
#include <SushiBLAS/graph_primitives/prefix_scan.hpp>
#include <SushiBLAS/graph_primitives/radix_plan.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using GraphPrimitives::counting_sort;
            using GraphPrimitives::prefix_scan;
            using SushiRuntime::LateCount;
            using SushiRuntime::Runtime;

            /// @brief Holds a value no sort of these tests writes.
            constexpr std::uint32_t K_CANARY = 0xDEADBEEFu;

            /// @brief Holds the nodes one radix digit records.
            constexpr std::size_t K_NODES_PER_DIGIT = 4;

            /** @brief Holds a stable host sort by key and the buckets it implies. */
            struct HostReference
            {
                std::vector<std::uint32_t> indices; /**< Element indices in stable key order. */
                std::vector<std::uint32_t> offsets; /**< key_range + 1 exclusive bucket starts. */
                std::vector<std::uint32_t> counts;  /**< key_range bucket sizes. */
            };

            /** @brief Returns the stable order of keys and the buckets of [0, key_range). */
            HostReference host_counting_sort(const std::vector<std::uint32_t>& keys,
                                             std::size_t key_range)
            {
                HostReference reference;
                reference.indices.resize(keys.size());
                std::iota(reference.indices.begin(), reference.indices.end(), std::uint32_t{0});
                std::stable_sort(reference.indices.begin(), reference.indices.end(),
                                 [&keys](std::uint32_t a, std::uint32_t b)
                                 {
                                     return keys[a] < keys[b];
                                 });

                reference.counts.assign(key_range, 0);
                for (std::uint32_t key : keys)
                    ++reference.counts[key];

                reference.offsets.assign(key_range + 1, 0);
                std::uint32_t running = 0;
                for (std::size_t k = 0; k < key_range; ++k)
                {
                    reference.offsets[k] = running;
                    running += reference.counts[k];
                }
                reference.offsets[key_range] = running;
                return reference;
            }

            /** @brief Returns the order std::sort leaves, which need not keep ties in place. */
            std::vector<std::uint32_t> unstable_order(const std::vector<std::uint32_t>& keys)
            {
                std::vector<std::uint32_t> order(keys.size());
                std::iota(order.begin(), order.end(), std::uint32_t{0});
                std::sort(order.begin(), order.end(),
                          [&keys](std::uint32_t a, std::uint32_t b)
                          {
                              return keys[a] < keys[b];
                          });
                return order;
            }

            /** @brief Returns count keys drawn from [0, key_range) with a fixed seed. */
            std::vector<std::uint32_t> seeded_keys(std::size_t count, std::size_t key_range,
                                                   unsigned seed)
            {
                std::mt19937 generator(seed);
                std::uniform_int_distribution<std::uint32_t> draw(
                    0, static_cast<std::uint32_t>(key_range - 1));
                std::vector<std::uint32_t> keys(count);
                for (std::uint32_t& key : keys)
                    key = draw(generator);
                return keys;
            }
        } // namespace

        /** @brief Tests stability against std::stable_sort on seeded duplicate keys. */
        TEST(Unit_GraphPrimitivesCountingSort, StableAgainstHostReferenceOnDuplicateKeys)
        {
            constexpr std::size_t K_COUNT = 4000;
            constexpr std::size_t K_KEY_RANGE = 32;

            const std::vector<std::uint32_t> host_keys =
                seeded_keys(K_COUNT, K_KEY_RANGE, 20260916);
            const HostReference reference = host_counting_sort(host_keys, K_KEY_RANGE);
            ASSERT_NE(unstable_order(host_keys), reference.indices)
                << "the input must exercise stability, or this test proves nothing";

            auto runtime = Runtime::create().value();
            auto keys = runtime.buffer<std::uint32_t>(K_COUNT).value();
            auto indices = runtime.buffer<std::uint32_t>(K_COUNT + 1).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_KEY_RANGE + 2).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            auto counts = runtime.buffer<std::uint32_t>(K_KEY_RANGE + 1).value();
            for (std::size_t i = 0; i < K_COUNT; ++i)
                keys[i] = host_keys[i];
            indices[K_COUNT] = K_CANARY;
            offsets[K_KEY_RANGE + 1] = K_CANARY;
            counts[K_KEY_RANGE] = K_CANARY;

            auto graph = runtime.graph();
            counting_sort(graph.recorder(), keys, indices, offsets, total,
                          LateCount::fixed(K_COUNT), LateCount::fixed(K_KEY_RANGE), &counts)
                .value();
            graph.run().value();

            EXPECT_EQ(total[0], K_COUNT);
            for (std::size_t i = 0; i < K_COUNT; ++i)
                ASSERT_EQ(indices[i], reference.indices[i]) << "i = " << i;
            for (std::size_t k = 0; k <= K_KEY_RANGE; ++k)
                ASSERT_EQ(offsets[k], reference.offsets[k]) << "k = " << k;
            for (std::size_t k = 0; k < K_KEY_RANGE; ++k)
                ASSERT_EQ(counts[k], reference.counts[k]) << "k = " << k;
            EXPECT_EQ(indices[K_COUNT], K_CANARY);
            EXPECT_EQ(offsets[K_KEY_RANGE + 1], K_CANARY);
            EXPECT_EQ(counts[K_KEY_RANGE], K_CANARY);
        }

        /** @brief Tests stability across digits when every key shares the same low byte. */
        TEST(Unit_GraphPrimitivesCountingSort, StableAcrossDigitsWhenTheLowDigitTies)
        {
            constexpr std::size_t K_COUNT = 4000;
            constexpr std::size_t K_KEY_CAPACITY = std::size_t{1} << 16;
            const std::vector<std::uint32_t> alphabet = {0x0100u, 0x0200u, 0x0300u, 0x0400u};

            std::mt19937 generator(20260917);
            std::uniform_int_distribution<std::size_t> pick(0, alphabet.size() - 1);
            std::vector<std::uint32_t> host_keys(K_COUNT);
            for (std::uint32_t& key : host_keys)
                key = alphabet[pick(generator)];

            const HostReference reference = host_counting_sort(host_keys, K_KEY_CAPACITY);
            ASSERT_NE(unstable_order(host_keys), reference.indices)
                << "the input must exercise stability, or this test proves nothing";

            auto runtime = Runtime::create().value();
            auto keys = runtime.buffer<std::uint32_t>(K_COUNT).value();
            auto indices = runtime.buffer<std::uint32_t>(K_COUNT).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_KEY_CAPACITY + 1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            for (std::size_t i = 0; i < K_COUNT; ++i)
                keys[i] = host_keys[i];

            auto graph = runtime.graph();
            counting_sort(graph.recorder(), keys, indices, offsets, total,
                          LateCount::fixed(K_COUNT), LateCount::fixed(K_KEY_CAPACITY)).value();
            graph.run().value();

            EXPECT_EQ(total[0], K_COUNT);
            for (std::size_t i = 0; i < K_COUNT; ++i)
                ASSERT_EQ(indices[i], reference.indices[i]) << "i = " << i;
            for (const std::uint32_t k : alphabet)
                ASSERT_EQ(offsets[k], reference.offsets[k]) << "k = " << k;
        }

        /** @brief Tests 10^6 seeded elements over a 2^20 key range against the host reference. */
        TEST(Unit_GraphPrimitivesCountingSort, LargeRandomCaseAgainstHostReference)
        {
            constexpr std::size_t K_COUNT = 1000000;
            constexpr std::size_t K_KEY_RANGE = std::size_t{1} << 20;

            const std::vector<std::uint32_t> host_keys =
                seeded_keys(K_COUNT, K_KEY_RANGE, 20260918);

            auto runtime = Runtime::create().value();
            auto keys = runtime.buffer<std::uint32_t>(K_COUNT).value();
            auto indices = runtime.buffer<std::uint32_t>(K_COUNT).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_KEY_RANGE + 1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            for (std::size_t i = 0; i < K_COUNT; ++i)
                keys[i] = host_keys[i];

            const HostReference reference = host_counting_sort(host_keys, K_KEY_RANGE);

            auto graph = runtime.graph();
            counting_sort(graph.recorder(), keys, indices, offsets, total,
                          LateCount::fixed(K_COUNT), LateCount::fixed(K_KEY_RANGE)).value();
            graph.run().value();

            EXPECT_EQ(total[0], K_COUNT);
            for (std::size_t i = 0; i < K_COUNT; ++i)
                ASSERT_EQ(indices[i], reference.indices[i]) << "i = " << i;
            for (std::size_t k = 0; k <= K_KEY_RANGE; ++k)
                ASSERT_EQ(offsets[k], reference.offsets[k]) << "k = " << k;
        }

        /** @brief Tests that equal keys keep the input order, since every element ties. */
        TEST(Unit_GraphPrimitivesCountingSort, AllKeysEqualKeepsInputOrder)
        {
            constexpr std::size_t K_COUNT = 777;
            constexpr std::size_t K_KEY_RANGE = 4;

            auto runtime = Runtime::create().value();
            auto keys = runtime.buffer<std::uint32_t>(K_COUNT).value();
            auto indices = runtime.buffer<std::uint32_t>(K_COUNT).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_KEY_RANGE + 1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            for (std::size_t i = 0; i < K_COUNT; ++i)
                keys[i] = 2;

            auto graph = runtime.graph();
            counting_sort(graph.recorder(), keys, indices, offsets, total,
                          LateCount::fixed(K_COUNT), LateCount::fixed(K_KEY_RANGE)).value();
            graph.run().value();

            for (std::size_t i = 0; i < K_COUNT; ++i)
                ASSERT_EQ(indices[i], i) << "i = " << i;
            EXPECT_EQ(offsets[2], 0u);
            EXPECT_EQ(offsets[3], K_COUNT);
            EXPECT_EQ(total[0], K_COUNT);
        }

        /** @brief Tests one element, the smallest input that runs every pass. */
        TEST(Unit_GraphPrimitivesCountingSort, OneElement)
        {
            auto runtime = Runtime::create().value();
            auto keys = runtime.buffer<std::uint32_t>(1).value();
            auto indices = runtime.buffer<std::uint32_t>(1).value();
            auto offsets = runtime.buffer<std::uint32_t>(9).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            keys[0] = 5;

            auto graph = runtime.graph();
            counting_sort(graph.recorder(), keys, indices, offsets, total, LateCount::fixed(1),
                          LateCount::fixed(8)).value();
            graph.run().value();

            EXPECT_EQ(indices[0], 0u);
            EXPECT_EQ(offsets[5], 0u);
            EXPECT_EQ(offsets[6], 1u);
            EXPECT_EQ(total[0], 1u);
        }

        /** @brief Tests that zero elements zero every bucket and write nothing past the window. */
        TEST(Unit_GraphPrimitivesCountingSort, ZeroElementsWritesZeroEverywhere)
        {
            constexpr std::size_t K_KEY_RANGE = 16;

            auto runtime = Runtime::create().value();
            auto keys = runtime.buffer<std::uint32_t>(1).value();
            auto indices = runtime.buffer<std::uint32_t>(1).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_KEY_RANGE + 2).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            auto counts = runtime.buffer<std::uint32_t>(K_KEY_RANGE + 1).value();
            for (std::size_t k = 0; k < offsets.size(); ++k)
                offsets[k] = K_CANARY;
            for (std::size_t k = 0; k < counts.size(); ++k)
                counts[k] = K_CANARY;
            total[0] = K_CANARY;

            auto graph = runtime.graph();
            counting_sort(graph.recorder(), keys, indices, offsets, total, LateCount::fixed(0),
                          LateCount::fixed(K_KEY_RANGE), &counts).value();
            graph.run().value();

            EXPECT_EQ(total[0], 0u);
            for (std::size_t k = 0; k <= K_KEY_RANGE; ++k)
                EXPECT_EQ(offsets[k], 0u) << "k = " << k;
            for (std::size_t k = 0; k < K_KEY_RANGE; ++k)
                EXPECT_EQ(counts[k], 0u) << "k = " << k;
            EXPECT_EQ(offsets[K_KEY_RANGE + 1], K_CANARY);
            EXPECT_EQ(counts[K_KEY_RANGE], K_CANARY);
        }

        /** @brief Tests the largest key of a 2^20 range, which sets bit 19. */
        TEST(Unit_GraphPrimitivesCountingSort, MaximumKeyOfA20BitRange)
        {
            constexpr std::size_t K_KEY_RANGE = std::size_t{1} << 20;
            const auto largest = static_cast<std::uint32_t>(K_KEY_RANGE - 1);
            const std::vector<std::uint32_t> host_keys = {largest, 0, largest, 1, 0};
            const std::size_t count = host_keys.size();

            auto runtime = Runtime::create().value();
            auto keys = runtime.buffer<std::uint32_t>(count).value();
            auto indices = runtime.buffer<std::uint32_t>(count).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_KEY_RANGE + 1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            for (std::size_t i = 0; i < count; ++i)
                keys[i] = host_keys[i];

            auto graph = runtime.graph();
            counting_sort(graph.recorder(), keys, indices, offsets, total,
                          LateCount::fixed(count), LateCount::fixed(K_KEY_RANGE)).value();
            graph.run().value();

            const HostReference reference = host_counting_sort(host_keys, K_KEY_RANGE);
            for (std::size_t i = 0; i < count; ++i)
                ASSERT_EQ(indices[i], reference.indices[i]) << "i = " << i;
            EXPECT_EQ(offsets[K_KEY_RANGE - 1], 3u);
            EXPECT_EQ(offsets[K_KEY_RANGE], 5u);
        }

        /** @brief Tests that each extra digit the key capacity needs records four more nodes. */
        TEST(Unit_GraphPrimitivesCountingSort, DigitCountFollowsKeyRangeCapacity)
        {
            const auto nodes_for = [](std::size_t key_capacity) -> std::size_t
            {
                constexpr std::size_t K_COUNT = 10;
                auto runtime = Runtime::create().value();
                auto keys = runtime.buffer<std::uint32_t>(K_COUNT).value();
                auto indices = runtime.buffer<std::uint32_t>(K_COUNT).value();
                auto offsets = runtime.buffer<std::uint32_t>(key_capacity + 1).value();
                auto total = runtime.buffer<std::uint32_t>(1).value();
                for (std::size_t i = 0; i < K_COUNT; ++i)
                    keys[i] = 0;

                auto graph = runtime.graph();
                graph.set_profiling(true);
                counting_sort(graph.recorder(), keys, indices, offsets, total,
                              LateCount::fixed(K_COUNT), LateCount::fixed(key_capacity)).value();
                const auto report = graph.run().value();
                EXPECT_EQ(report.node_events.size(), graph.recorder().node_count());
                return report.node_events.size();
            };

            const std::size_t two_digits = nodes_for(std::size_t{1} << 16);
            const std::size_t three_digits = nodes_for(std::size_t{1} << 20);
            EXPECT_EQ(three_digits - two_digits, K_NODES_PER_DIGIT)
                << "2^20 needs three digit passes, 2^16 needs two";
            EXPECT_EQ(two_digits, 2 * K_NODES_PER_DIGIT + 2)
                << "two digits, the resolver and the offsets pass";
        }

        /** @brief Tests that a count and a key range that change between runs keep one compile. */
        TEST(Unit_GraphPrimitivesCountingSort, LateBoundChangingCountAndKeyRangeKeepsOneCompile)
        {
            constexpr std::size_t K_COUNT_CAPACITY = 2000;
            constexpr std::size_t K_KEY_CAPACITY = std::size_t{1} << 16;

            auto runtime = Runtime::create().value();
            auto keys = runtime.buffer<std::uint32_t>(K_COUNT_CAPACITY).value();
            auto indices = runtime.buffer<std::uint32_t>(K_COUNT_CAPACITY).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_KEY_CAPACITY + 1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();

            std::size_t live_n = 10;
            std::size_t live_key_range = 4;
            const LateCount count =
                LateCount::from_host(K_COUNT_CAPACITY, [&live_n]() { return live_n; });
            const LateCount key_range = LateCount::from_host(
                K_KEY_CAPACITY, [&live_key_range]() { return live_key_range; });

            auto graph = runtime.graph();
            counting_sort(graph.recorder(), keys, indices, offsets, total, count, key_range)
                .value();

            const auto run_and_check = [&](std::size_t n, std::size_t range)
            {
                live_n = n;
                live_key_range = range;
                const std::vector<std::uint32_t> host_keys =
                    seeded_keys(n, range, static_cast<unsigned>(n * 31 + range));
                for (std::size_t i = 0; i < n; ++i)
                    keys[i] = host_keys[i];

                graph.run().value();

                const HostReference reference = host_counting_sort(host_keys, range);
                for (std::size_t i = 0; i < n; ++i)
                    ASSERT_EQ(indices[i], reference.indices[i])
                        << "n = " << n << ", key_range = " << range << ", i = " << i;
                EXPECT_EQ(total[0], n);
                for (std::size_t k = 0; k <= range; ++k)
                    ASSERT_EQ(offsets[k], reference.offsets[k])
                        << "n = " << n << ", key_range = " << range << ", k = " << k;
            };

            run_and_check(10, 4);
            run_and_check(1500, 60000);
            run_and_check(3, 2);

            EXPECT_EQ(graph.compile_count(), 1u)
                << "a live count or key range change between runs must not trigger a recompile";
        }

        /** @brief Tests that live key ranges of one, two and three digits all end in indices. */
        TEST(Unit_GraphPrimitivesCountingSort, LiveKeyRangeBelowCapacityLandsInIndices)
        {
            constexpr std::size_t K_COUNT = 3000;
            constexpr std::size_t K_KEY_CAPACITY = std::size_t{1} << 24;

            auto runtime = Runtime::create().value();
            auto keys = runtime.buffer<std::uint32_t>(K_COUNT).value();
            auto indices = runtime.buffer<std::uint32_t>(K_COUNT).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_KEY_CAPACITY + 1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();

            std::size_t live_key_range = 0;
            const LateCount key_range = LateCount::from_host(
                K_KEY_CAPACITY, [&live_key_range]() { return live_key_range; });

            auto graph = runtime.graph();
            counting_sort(graph.recorder(), keys, indices, offsets, total,
                          LateCount::fixed(K_COUNT), key_range).value();

            for (std::size_t range : {std::size_t{200}, std::size_t{40000}, std::size_t{5000000},
                                      std::size_t{3}})
            {
                live_key_range = range;
                const std::vector<std::uint32_t> host_keys =
                    seeded_keys(K_COUNT, range, static_cast<unsigned>(range));
                for (std::size_t i = 0; i < K_COUNT; ++i)
                    keys[i] = host_keys[i];

                graph.run().value();

                const HostReference reference = host_counting_sort(host_keys, range);
                for (std::size_t i = 0; i < K_COUNT; ++i)
                    ASSERT_EQ(indices[i], reference.indices[i])
                        << "key_range = " << range << ", i = " << i;
                for (std::size_t k = 0; k <= range; ++k)
                    ASSERT_EQ(offsets[k], reference.offsets[k])
                        << "key_range = " << range << ", k = " << k;
            }
            EXPECT_EQ(graph.compile_count(), 1u);
        }

        /** @brief Tests that the digits above the live key range do not run. */
        TEST(Unit_GraphPrimitivesCountingSort, DigitsAboveTheLiveKeyRangeDoNotRun)
        {
            constexpr std::size_t K_COUNT = 4;
            constexpr std::size_t K_KEY_CAPACITY = std::size_t{1} << 16;

            auto runtime = Runtime::create().value();
            auto keys = runtime.buffer<std::uint32_t>(K_COUNT).value();
            auto indices = runtime.buffer<std::uint32_t>(K_COUNT).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_KEY_CAPACITY + 1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();

            // One digit puts 0x0101 between 0 and 2; a second digit would put it last.
            keys[0] = 0x0101;
            keys[1] = 2;
            keys[2] = 0;
            keys[3] = 3;

            auto graph = runtime.graph();
            counting_sort(graph.recorder(), keys, indices, offsets, total,
                          LateCount::fixed(K_COUNT),
                          LateCount::from_host(K_KEY_CAPACITY, []() { return std::size_t{4}; }))
                .value();
            graph.run().value();

            const std::vector<std::uint32_t> one_digit_order = {2, 0, 1, 3};
            for (std::size_t i = 0; i < K_COUNT; ++i)
                EXPECT_EQ(indices[i], one_digit_order[i]) << "i = " << i;
        }

        /** @brief Tests that a DynamicGraph region records the sort and the scan as a Graph. */
        TEST(Unit_GraphPrimitivesCountingSort, RegionsRecordScanAndCountingSort)
        {
            constexpr std::size_t K_COUNT_CAPACITY = 700;
            constexpr std::size_t K_KEY_CAPACITY = 1000;

            auto runtime = Runtime::create().value();
            auto keys = runtime.buffer<std::uint32_t>(K_COUNT_CAPACITY).value();
            auto indices = runtime.buffer<std::uint32_t>(K_COUNT_CAPACITY).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_KEY_CAPACITY + 1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            auto scan_offsets = runtime.buffer<std::uint32_t>(K_COUNT_CAPACITY + 1).value();
            auto scan_total = runtime.buffer<std::uint32_t>(1).value();

            std::size_t live_n = 0;
            std::size_t live_key_range = 0;
            const LateCount count =
                LateCount::from_host(K_COUNT_CAPACITY, [&live_n]() { return live_n; });
            const LateCount key_range = LateCount::from_host(
                K_KEY_CAPACITY, [&live_key_range]() { return live_key_range; });

            auto world = runtime.dynamic_graph();
            counting_sort(world.region(1).recorder(), keys, indices, offsets, total, count,
                          key_range).value();
            prefix_scan(world.region(2).recorder(), keys, scan_offsets, scan_total, count).value();

            for (const auto& live : {std::pair<std::size_t, std::size_t>{300, 200},
                                     std::pair<std::size_t, std::size_t>{700, 1000}})
            {
                live_n = live.first;
                live_key_range = live.second;
                const std::vector<std::uint32_t> host_keys =
                    seeded_keys(live_n, live_key_range, static_cast<unsigned>(live_n));
                for (std::size_t i = 0; i < live_n; ++i)
                    keys[i] = host_keys[i];

                world.run().value();

                const HostReference reference = host_counting_sort(host_keys, live_key_range);
                for (std::size_t i = 0; i < live_n; ++i)
                    ASSERT_EQ(indices[i], reference.indices[i])
                        << "n = " << live_n << ", i = " << i;
                for (std::size_t k = 0; k <= live_key_range; ++k)
                    ASSERT_EQ(offsets[k], reference.offsets[k])
                        << "n = " << live_n << ", k = " << k;
                std::uint32_t running = 0;
                for (std::size_t i = 0; i < live_n; ++i)
                {
                    ASSERT_EQ(scan_offsets[i], running) << "n = " << live_n << ", i = " << i;
                    running += host_keys[i];
                }
                EXPECT_EQ(scan_offsets[live_n], running);
                EXPECT_EQ(scan_total[0], running);
            }
        }

        /** @brief Tests that each run sorts the keys as they stand when it runs. */
        TEST(Unit_GraphPrimitivesCountingSort, EachRunSortsItsOwnKeys)
        {
            constexpr std::size_t K_COUNT = 500;
            constexpr std::size_t K_KEY_RANGE = 20;
            constexpr std::uint32_t K_STEPS = 4;

            auto runtime = Runtime::create().value();
            auto keys = runtime.buffer<std::uint32_t>(K_COUNT).value();
            auto indices = runtime.buffer<std::uint32_t>(K_COUNT).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_KEY_RANGE + 1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();

            auto graph = runtime.graph();
            counting_sort(graph.recorder(), keys, indices, offsets, total,
                          LateCount::fixed(K_COUNT), LateCount::fixed(K_KEY_RANGE)).value();

            for (std::uint32_t step = 0; step < K_STEPS; ++step)
            {
                std::vector<std::uint32_t> host_keys(K_COUNT);
                for (std::size_t i = 0; i < K_COUNT; ++i)
                {
                    host_keys[i] = static_cast<std::uint32_t>((i * 7 + step * 3) % K_KEY_RANGE);
                    keys[i] = host_keys[i];
                }

                graph.run().value();

                const HostReference reference = host_counting_sort(host_keys, K_KEY_RANGE);
                for (std::size_t i = 0; i < K_COUNT; ++i)
                    ASSERT_EQ(indices[i], reference.indices[i])
                        << "step " << step << ", i = " << i;
                for (std::size_t k = 0; k <= K_KEY_RANGE; ++k)
                    ASSERT_EQ(offsets[k], reference.offsets[k])
                        << "step " << step << ", k = " << k;
            }
        }

        /** @brief Tests that every node of one call shares one node_index on every step. */
        TEST(Unit_GraphPrimitivesCountingSort, SharesOneNodeIndexAcrossStepsAndPasses)
        {
            // One digit records the resolver, four digit passes, the offsets and the counts.
            constexpr std::size_t K_COUNT_CAPACITY = 300;
            constexpr std::size_t K_KEY_CAPACITY = 50;
            constexpr std::size_t K_EXPECTED_PASSES = K_NODES_PER_DIGIT + 3;
            constexpr std::size_t K_STEPS = 4;

            auto runtime = Runtime::create().value();
            auto keys = runtime.buffer<std::uint32_t>(K_COUNT_CAPACITY).value();
            auto indices = runtime.buffer<std::uint32_t>(K_COUNT_CAPACITY).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_KEY_CAPACITY + 1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            auto counts = runtime.buffer<std::uint32_t>(K_KEY_CAPACITY).value();
            for (std::size_t i = 0; i < K_COUNT_CAPACITY; ++i)
                keys[i] = static_cast<std::uint32_t>(i % K_KEY_CAPACITY);

            auto graph = runtime.graph();
            graph.set_profiling(true);
            counting_sort(graph.recorder(), keys, indices, offsets, total,
                          LateCount::fixed(K_COUNT_CAPACITY), LateCount::fixed(K_KEY_CAPACITY),
                          &counts).value();
            EXPECT_EQ(graph.recorder().rule_count(), 1u);

            const auto report = graph.run(K_STEPS).value();

            ASSERT_EQ(report.node_events.size(), K_EXPECTED_PASSES * K_STEPS);

            const std::uint32_t node_index = report.node_events.front().node_index;
            std::vector<std::size_t> per_invocation(K_STEPS, 0);
            for (const auto& record : report.node_events)
            {
                EXPECT_EQ(record.node_index, node_index)
                    << "every pass of one counting_sort() call must share one node_index";
                ASSERT_LT(record.invocation, K_STEPS);
                ++per_invocation[record.invocation];
            }
            for (std::size_t step = 0; step < K_STEPS; ++step)
                EXPECT_EQ(per_invocation[step], K_EXPECTED_PASSES) << "step " << step;
        }

        /** @brief Tests that each node launches at the width its pass has in the plan. */
        TEST(Unit_GraphPrimitivesCountingSort, NodeWidthsMatchThePlan)
        {
            // Resolver, histogram, column scan, bin base, placement, offsets, counts.
            constexpr std::size_t K_COUNT_CAPACITY = 300;
            constexpr std::size_t K_KEY_CAPACITY = 50;
            const auto bins =
                static_cast<std::uint32_t>(GraphPrimitives::Detail::K_RADIX_DIGIT_BINS);
            const std::vector<std::uint32_t> expected_widths = {1u, 2u, bins, 1u, 2u, 51u, 50u};

            auto runtime = Runtime::create().value();
            auto keys = runtime.buffer<std::uint32_t>(K_COUNT_CAPACITY).value();
            auto indices = runtime.buffer<std::uint32_t>(K_COUNT_CAPACITY).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_KEY_CAPACITY + 1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            auto counts = runtime.buffer<std::uint32_t>(K_KEY_CAPACITY).value();
            for (std::size_t i = 0; i < K_COUNT_CAPACITY; ++i)
                keys[i] = static_cast<std::uint32_t>(i % K_KEY_CAPACITY);

            auto graph = runtime.graph();
            graph.set_profiling(true);
            counting_sort(graph.recorder(), keys, indices, offsets, total,
                          LateCount::fixed(K_COUNT_CAPACITY), LateCount::fixed(K_KEY_CAPACITY),
                          &counts).value();

            const auto report = graph.run().value();

            ASSERT_EQ(report.node_events.size(), expected_widths.size());
            for (std::size_t i = 0; i < expected_widths.size(); ++i)
            {
                EXPECT_EQ(report.node_events[i].launched_width, expected_widths[i])
                    << "pass " << i;
                EXPECT_EQ(report.node_events[i].live_width, expected_widths[i]) << "pass " << i;
            }
        }

        /** @brief Tests that broken size rules are refused with invalid_graph. */
        TEST(Unit_GraphPrimitivesCountingSort, RejectsMisSizedArguments)
        {
            using Words = SushiRuntime::Buffer<std::uint32_t>;

            auto runtime = Runtime::create().value();
            auto keys = runtime.buffer<std::uint32_t>(8).value();
            auto indices = runtime.buffer<std::uint32_t>(8).value();
            auto offsets = runtime.buffer<std::uint32_t>(5).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            auto small = runtime.buffer<std::uint32_t>(3).value();
            auto empty = runtime.buffer<std::uint32_t>(0).value();

            auto graph = runtime.graph();
            const auto refused_code = [&graph, &keys](const Words& i, const Words& o,
                                                      const Words& t, std::size_t n,
                                                      std::size_t range, const Words* c)
            {
                const SushiRuntime::Result<void> recorded =
                    counting_sort(graph.recorder(), keys, i, o, t, LateCount::fixed(n),
                                  LateCount::fixed(range), c);
                return recorded ? SushiRuntime::ErrorCode::Ok : recorded.error().code;
            };
            const auto invalid = SushiRuntime::ErrorCode::InvalidGraph;

            EXPECT_EQ(refused_code(indices, offsets, total, 8, 0, nullptr), invalid)
                << "a key range capacity of zero has no bucket";
            EXPECT_EQ(refused_code(indices, offsets, total, 9, 4, nullptr), invalid)
                << "nine keys do not fit a buffer of eight";
            EXPECT_EQ(refused_code(small, offsets, total, 8, 4, nullptr), invalid)
                << "three indices cannot hold eight elements";
            EXPECT_EQ(refused_code(indices, small, total, 8, 4, nullptr), invalid)
                << "three offsets cannot hold four buckets and the sentinel";
            EXPECT_EQ(refused_code(indices, offsets, empty, 8, 4, nullptr), invalid)
                << "an empty total cannot receive the count";
            EXPECT_EQ(refused_code(indices, offsets, total, 8, 4, &small), invalid)
                << "three counts cannot hold four buckets";
            EXPECT_EQ(graph.recorder().node_count(), 0u);
        }

        /** @brief Tests that a live element count above its capacity fails the run. */
        TEST(Unit_GraphPrimitivesCountingSort, LateBoundCountAboveCapacityFailsTheRun)
        {
            auto runtime = Runtime::create().value();
            auto keys = runtime.buffer<std::uint32_t>(100).value();
            auto indices = runtime.buffer<std::uint32_t>(100).value();
            auto offsets = runtime.buffer<std::uint32_t>(17).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();

            auto graph = runtime.graph();
            counting_sort(graph.recorder(), keys, indices, offsets, total,
                          LateCount::from_host(100, []() { return std::size_t{101}; }),
                          LateCount::fixed(16)).value();
            EXPECT_FALSE(graph.run().has_value());
        }

        /** @brief Tests that a live key range above its capacity fails the run. */
        TEST(Unit_GraphPrimitivesCountingSort, LateBoundKeyRangeAboveCapacityFailsTheRun)
        {
            auto runtime = Runtime::create().value();
            auto keys = runtime.buffer<std::uint32_t>(100).value();
            auto indices = runtime.buffer<std::uint32_t>(100).value();
            auto offsets = runtime.buffer<std::uint32_t>(17).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();

            auto graph = runtime.graph();
            counting_sort(graph.recorder(), keys, indices, offsets, total, LateCount::fixed(100),
                          LateCount::from_host(16, []() { return std::size_t{17}; })).value();
            EXPECT_FALSE(graph.run().has_value());
        }

    } // namespace Test
} // namespace SushiBLAS
