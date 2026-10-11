/**************************************************************************/
/* test_prefix_scan.cpp                                                   */
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
 * @file test_prefix_scan.cpp
 * @brief Tests prefix_scan() against a host scan, across live counts, replays and node widths.
 * @author Mustafa Garip
 * @see modules/foundation/graph_primitives/tests/README.md
 */

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/graph/graph.hpp>
#include <SushiRuntime/api/runtime/runtime.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>
#include <SushiRuntime/core/error.hpp>

#include <SushiBLAS/graph_primitives/fixed_order_reduce.hpp>
#include <SushiBLAS/graph_primitives/prefix_scan.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using GraphPrimitives::prefix_scan;
            using SushiRuntime::LateCount;
            using SushiRuntime::Runtime;
            using SushiRuntime::SushiException;

            /// @brief Holds the tile size the level plan of a scan is built on.
            constexpr std::size_t K_TILE = GraphPrimitives::K_REDUCE_TILE;

            /// @brief Holds a value no scan of these tests produces.
            constexpr std::uint32_t K_CANARY = 0xDEADBEEFu;

            /// @brief Holds a capacity of two levels: three tiles under one top tile.
            constexpr std::size_t K_TWO_LEVEL_CAPACITY = 2 * K_TILE + 88;

            /**
             * @brief Writes the exclusive prefix sums of the first n counts, modulo 2^32.
             * @param offsets Holds at least n + 1 elements; offsets[n] receives the sum.
             * @return The sum of the first n counts.
             */
            std::uint32_t host_exclusive_scan(const std::vector<std::uint32_t>& counts,
                                              std::size_t n, std::vector<std::uint32_t>& offsets)
            {
                std::uint32_t running = 0;
                for (std::size_t i = 0; i < n; ++i)
                {
                    offsets[i] = running;
                    running += counts[i];
                }
                offsets[n] = running;
                return running;
            }

            /** @brief Returns n counts that cycle through 1 to 13. */
            std::vector<std::uint32_t> cycling_counts(std::size_t n)
            {
                std::vector<std::uint32_t> counts(n);
                for (std::size_t i = 0; i < n; ++i)
                    counts[i] = static_cast<std::uint32_t>((i % 13) + 1);
                return counts;
            }
        } // namespace

        /** @brief Tests exact agreement with the host scan at the tile and level edges. */
        TEST(Unit_GraphPrimitivesPrefixScan, AgreesWithTheHostScanAtTileEdges)
        {
            const std::vector<std::size_t> sizes = {0, 1, K_TILE - 1, K_TILE, K_TILE + 1,
                                                    K_TILE * K_TILE, K_TILE * K_TILE + 1,
                                                    1000000};

            for (std::size_t n : sizes)
            {
                auto runtime = Runtime::create();
                auto counts = runtime.buffer<std::uint32_t>(n == 0 ? 1 : n);
                auto offsets = runtime.buffer<std::uint32_t>(n + 2);
                auto total = runtime.buffer<std::uint32_t>(1);

                const std::vector<std::uint32_t> host_counts = cycling_counts(n);
                for (std::size_t i = 0; i < n; ++i)
                    counts[i] = host_counts[i];
                for (std::size_t i = 0; i < offsets.size(); ++i)
                    offsets[i] = K_CANARY;
                total[0] = K_CANARY;

                std::vector<std::uint32_t> expected_offsets(n + 1, 0);
                const std::uint32_t expected_total =
                    host_exclusive_scan(host_counts, n, expected_offsets);

                auto graph = runtime.graph();
                prefix_scan(graph.recorder(), counts, offsets, total, LateCount::fixed(n));
                graph.run();

                EXPECT_EQ(total[0], expected_total) << "n = " << n;
                for (std::size_t i = 0; i <= n; ++i)
                    ASSERT_EQ(offsets[i], expected_offsets[i]) << "n = " << n << ", i = " << i;
                EXPECT_EQ(offsets[n + 1], K_CANARY)
                    << "n = " << n << ": a write reached past the declared window";
            }
        }

        /** @brief Tests that offsets[n] equals the total the scan publishes. */
        TEST(Unit_GraphPrimitivesPrefixScan, OffsetsAtLiveCountEqualsTotal)
        {
            constexpr std::size_t K_COUNT = K_TILE + 1;

            auto runtime = Runtime::create();
            auto counts = runtime.buffer<std::uint32_t>(K_COUNT);
            auto offsets = runtime.buffer<std::uint32_t>(K_COUNT + 1);
            auto total = runtime.buffer<std::uint32_t>(1);
            for (std::size_t i = 0; i < K_COUNT; ++i)
                counts[i] = static_cast<std::uint32_t>(i + 1);

            auto graph = runtime.graph();
            prefix_scan(graph.recorder(), counts, offsets, total, LateCount::fixed(K_COUNT));
            graph.run();

            EXPECT_EQ(total[0], static_cast<std::uint32_t>(K_COUNT * (K_COUNT + 1) / 2));
            EXPECT_EQ(offsets[K_COUNT], total[0]);
        }

        /** @brief Tests that a live count that changes between runs keeps one compile. */
        TEST(Unit_GraphPrimitivesPrefixScan, LateBoundChangingCountKeepsOneCompile)
        {
            constexpr std::size_t K_CAPACITY = 1000003;

            auto runtime = Runtime::create();
            auto counts = runtime.buffer<std::uint32_t>(K_CAPACITY);
            auto offsets = runtime.buffer<std::uint32_t>(K_CAPACITY + 1);
            auto total = runtime.buffer<std::uint32_t>(1);
            for (std::size_t i = 0; i < K_CAPACITY; ++i)
                counts[i] = 1;

            std::size_t live_n = 1000;
            const LateCount count =
                LateCount::from_host(K_CAPACITY, [&live_n]() { return live_n; });

            auto graph = runtime.graph();
            prefix_scan(graph.recorder(), counts, offsets, total, count);

            graph.run();
            EXPECT_EQ(total[0], 1000u);
            EXPECT_EQ(offsets[1000], 1000u);

            live_n = 700000;
            graph.run();
            EXPECT_EQ(total[0], 700000u);
            EXPECT_EQ(offsets[700000], 700000u);

            live_n = 3;
            graph.run();
            EXPECT_EQ(total[0], 3u);
            EXPECT_EQ(offsets[3], 3u);

            EXPECT_EQ(graph.compile_count(), 1u)
                << "a live count change between runs must not trigger a recompile";
        }

        /** @brief Tests that each run scans the counts the buffer holds when it runs. */
        TEST(Unit_GraphPrimitivesPrefixScan, EachRunScansItsOwnCounts)
        {
            constexpr std::size_t K_CAPACITY = 300;

            auto runtime = Runtime::create();
            auto counts = runtime.buffer<std::uint32_t>(K_CAPACITY);
            auto offsets = runtime.buffer<std::uint32_t>(K_CAPACITY + 1);
            auto total = runtime.buffer<std::uint32_t>(1);

            auto graph = runtime.graph();
            prefix_scan(graph.recorder(), counts, offsets, total, LateCount::fixed(K_CAPACITY));

            for (std::uint32_t step = 0; step < 4; ++step)
            {
                std::vector<std::uint32_t> host_counts(K_CAPACITY);
                for (std::size_t i = 0; i < K_CAPACITY; ++i)
                {
                    host_counts[i] = static_cast<std::uint32_t>((i + step) % 5);
                    counts[i] = host_counts[i];
                }

                std::vector<std::uint32_t> expected_offsets(K_CAPACITY + 1, 0);
                const std::uint32_t expected_total =
                    host_exclusive_scan(host_counts, K_CAPACITY, expected_offsets);

                graph.run();

                EXPECT_EQ(total[0], expected_total) << "step " << step;
                for (std::size_t i = 0; i <= K_CAPACITY; ++i)
                    ASSERT_EQ(offsets[i], expected_offsets[i])
                        << "step " << step << ", i = " << i;
            }

            EXPECT_EQ(graph.compile_count(), 1u);
        }

        /** @brief Tests that every node of one call shares one node_index on every step. */
        TEST(Unit_GraphPrimitivesPrefixScan, SharesOneNodeIndexAcrossStepsAndPasses)
        {
            // Two levels record a resolver, one up pass, two down passes and the closing write.
            constexpr std::size_t K_EXPECTED_PASSES = 5;
            constexpr std::size_t K_STEPS = 4;

            auto runtime = Runtime::create();
            auto counts = runtime.buffer<std::uint32_t>(K_TWO_LEVEL_CAPACITY);
            auto offsets = runtime.buffer<std::uint32_t>(K_TWO_LEVEL_CAPACITY + 1);
            auto total = runtime.buffer<std::uint32_t>(1);
            for (std::size_t i = 0; i < K_TWO_LEVEL_CAPACITY; ++i)
                counts[i] = static_cast<std::uint32_t>((i % 7) + 1);

            auto graph = runtime.graph();
            graph.set_profiling(true);
            prefix_scan(graph.recorder(), counts, offsets, total,
                        LateCount::fixed(K_TWO_LEVEL_CAPACITY));
            EXPECT_EQ(graph.recorder().node_count(), K_EXPECTED_PASSES);
            EXPECT_EQ(graph.recorder().rule_count(), 1u);

            const auto report = graph.run(K_STEPS);

            ASSERT_EQ(report.node_events.size(), K_EXPECTED_PASSES * K_STEPS);

            const std::uint32_t node_index = report.node_events.front().node_index;
            std::vector<std::size_t> per_invocation(K_STEPS, 0);
            for (const auto& record : report.node_events)
            {
                EXPECT_EQ(record.node_index, node_index)
                    << "every pass of one prefix_scan() call must share one node_index";
                ASSERT_LT(record.invocation, K_STEPS);
                ++per_invocation[record.invocation];
            }
            for (std::size_t step = 0; step < K_STEPS; ++step)
                EXPECT_EQ(per_invocation[step], K_EXPECTED_PASSES) << "step " << step;
        }

        /** @brief Tests that each node launches at the width its pass has in the level plan. */
        TEST(Unit_GraphPrimitivesPrefixScan, NodeWidthsMatchTheLevelPlan)
        {
            // The nodes are the resolver, up 0, the top down pass, down 0 and the closing write.
            const std::vector<std::uint32_t> expected_widths = {1u, 3u, 1u, 3u, 1u};

            auto runtime = Runtime::create();
            auto counts = runtime.buffer<std::uint32_t>(K_TWO_LEVEL_CAPACITY);
            auto offsets = runtime.buffer<std::uint32_t>(K_TWO_LEVEL_CAPACITY + 1);
            auto total = runtime.buffer<std::uint32_t>(1);
            for (std::size_t i = 0; i < K_TWO_LEVEL_CAPACITY; ++i)
                counts[i] = static_cast<std::uint32_t>((i % 7) + 1);

            auto graph = runtime.graph();
            graph.set_profiling(true);
            prefix_scan(graph.recorder(), counts, offsets, total,
                        LateCount::fixed(K_TWO_LEVEL_CAPACITY));

            const auto report = graph.run();

            ASSERT_EQ(report.node_events.size(), expected_widths.size());
            for (std::size_t i = 0; i < expected_widths.size(); ++i)
            {
                EXPECT_EQ(report.node_events[i].launched_width, expected_widths[i])
                    << "pass " << i;
                EXPECT_EQ(report.node_events[i].live_width, expected_widths[i]) << "pass " << i;
            }
        }

        /** @brief Tests that a capacity of zero writes a zero total and a zero offsets[0]. */
        TEST(Unit_GraphPrimitivesPrefixScan, ZeroCapacityWritesZeroTotal)
        {
            auto runtime = Runtime::create();
            auto counts = runtime.buffer<std::uint32_t>(1);
            auto offsets = runtime.buffer<std::uint32_t>(1);
            auto total = runtime.buffer<std::uint32_t>(1);
            offsets[0] = K_CANARY;
            total[0] = K_CANARY;

            auto graph = runtime.graph();
            prefix_scan(graph.recorder(), counts, offsets, total, LateCount::fixed(0));
            graph.run();

            EXPECT_EQ(offsets[0], 0u);
            EXPECT_EQ(total[0], 0u);
        }

        /** @brief Tests that a short buffer is refused with invalid_graph and records nothing. */
        TEST(Unit_GraphPrimitivesPrefixScan, RejectsMisSizedArguments)
        {
            using Words = SushiRuntime::Buffer<std::uint32_t>;

            auto runtime = Runtime::create();
            auto counts = runtime.buffer<std::uint32_t>(8);
            auto offsets = runtime.buffer<std::uint32_t>(9);
            auto total = runtime.buffer<std::uint32_t>(1);
            auto empty = runtime.buffer<std::uint32_t>(0);

            auto graph = runtime.graph();
            const auto refused_code = [&graph](const Words& c, const Words& o, const Words& t,
                                               std::size_t capacity)
            {
                try
                {
                    prefix_scan(graph.recorder(), c, o, t, LateCount::fixed(capacity));
                }
                catch (const SushiException& refused)
                {
                    return refused.error().code;
                }
                return SushiRuntime::Errc::ok;
            };

            EXPECT_EQ(refused_code(counts, offsets, total, 9),
                      SushiRuntime::Errc::invalid_graph)
                << "nine counts do not fit a buffer of eight";
            EXPECT_EQ(refused_code(counts, counts, total, 8),
                      SushiRuntime::Errc::invalid_graph)
                << "eight offsets cannot hold the sentinel of eight counts";
            EXPECT_EQ(refused_code(counts, offsets, empty, 8),
                      SushiRuntime::Errc::invalid_graph)
                << "an empty total cannot receive the sum";
            EXPECT_EQ(graph.recorder().node_count(), 0u);
        }

        /** @brief Tests that a live count above its capacity fails the run. */
        TEST(Unit_GraphPrimitivesPrefixScan, LateBoundAboveCapacityFailsTheRun)
        {
            auto runtime = Runtime::create();
            auto counts = runtime.buffer<std::uint32_t>(100);
            auto offsets = runtime.buffer<std::uint32_t>(101);
            auto total = runtime.buffer<std::uint32_t>(1);

            auto graph = runtime.graph();
            prefix_scan(graph.recorder(), counts, offsets, total,
                        LateCount::from_host(100, []() { return std::size_t{101}; }));
            EXPECT_THROW(graph.run(), SushiException);
        }

    } // namespace Test
} // namespace SushiBLAS
