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
 * @see tests/foundation/graph_primitives/README.md
 */

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/graph/graph.hpp>
#include <SushiRuntime/api/runtime/runtime.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>
#include <SushiRuntime/core/error.hpp>

#include <SushiBLAS/graph_primitives/fixed_order_reduce.hpp>
#include <SushiBLAS/graph_primitives/prefix_scan.hpp>
#include <SushiBLAS/graph_primitives/primitive_options.hpp>
#include "device_sized_harness.hpp"
#include "plan_labels.hpp"
#include "window_probe.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using GraphPrimitives::prefix_scan;
            using SushiRuntime::LateCount;
            using SushiRuntime::Runtime;

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
        TEST(Medium_GraphPrimitivesPrefixScan, AgreesWithTheHostScanAtTileEdges)
        {
            const std::vector<std::size_t> sizes = {0, 1, K_TILE - 1, K_TILE, K_TILE + 1,
                                                    K_TILE * K_TILE, K_TILE * K_TILE + 1,
                                                    1000000};

            for (std::size_t n : sizes)
            {
                auto runtime = Runtime::create().value();
                auto counts = runtime.buffer<std::uint32_t>(n == 0 ? 1 : n).value();
                auto offsets = runtime.buffer<std::uint32_t>(n + 2).value();
                auto total = runtime.buffer<std::uint32_t>(1).value();

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
                prefix_scan(graph.recorder(), counts, offsets, total, LateCount::fixed(n)).value();
                graph.run().value();

                EXPECT_EQ(total[0], expected_total) << "n = " << n;
                for (std::size_t i = 0; i <= n; ++i)
                    ASSERT_EQ(offsets[i], expected_offsets[i]) << "n = " << n << ", i = " << i;
                EXPECT_EQ(offsets[n + 1], K_CANARY)
                    << "n = " << n << ": a write reached past the declared window";
            }
        }

        /** @brief Tests that offsets[n] equals the total the scan publishes. */
        TEST(Medium_GraphPrimitivesPrefixScan, OffsetsAtLiveCountEqualsTotal)
        {
            constexpr std::size_t K_COUNT = K_TILE + 1;

            auto runtime = Runtime::create().value();
            auto counts = runtime.buffer<std::uint32_t>(K_COUNT).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_COUNT + 1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            for (std::size_t i = 0; i < K_COUNT; ++i)
                counts[i] = static_cast<std::uint32_t>(i + 1);

            auto graph = runtime.graph();
            prefix_scan(graph.recorder(), counts, offsets, total, LateCount::fixed(K_COUNT))
                .value();
            graph.run().value();

            EXPECT_EQ(total[0], static_cast<std::uint32_t>(K_COUNT * (K_COUNT + 1) / 2));
            EXPECT_EQ(offsets[K_COUNT], total[0]);
        }

        /** @brief Tests that a live count that changes between runs keeps one compile. */
        TEST(Medium_GraphPrimitivesPrefixScan, LateBoundChangingCountKeepsOneCompile)
        {
            constexpr std::size_t K_CAPACITY = 1000003;

            auto runtime = Runtime::create().value();
            auto counts = runtime.buffer<std::uint32_t>(K_CAPACITY).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_CAPACITY + 1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            for (std::size_t i = 0; i < K_CAPACITY; ++i)
                counts[i] = 1;

            std::size_t live_n = 1000;
            const LateCount count =
                LateCount::from_host(K_CAPACITY, [&live_n]() { return live_n; });

            auto graph = runtime.graph();
            prefix_scan(graph.recorder(), counts, offsets, total, count).value();

            graph.run().value();
            EXPECT_EQ(total[0], 1000u);
            EXPECT_EQ(offsets[1000], 1000u);

            live_n = 700000;
            graph.run().value();
            EXPECT_EQ(total[0], 700000u);
            EXPECT_EQ(offsets[700000], 700000u);

            live_n = 3;
            graph.run().value();
            EXPECT_EQ(total[0], 3u);
            EXPECT_EQ(offsets[3], 3u);

            EXPECT_EQ(graph.compile_count(), 1u)
                << "a live count change between runs must not trigger a recompile";
        }

        /** @brief Tests that each run scans the counts the buffer holds when it runs. */
        TEST(Medium_GraphPrimitivesPrefixScan, EachRunScansItsOwnCounts)
        {
            constexpr std::size_t K_CAPACITY = 300;

            auto runtime = Runtime::create().value();
            auto counts = runtime.buffer<std::uint32_t>(K_CAPACITY).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_CAPACITY + 1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();

            auto graph = runtime.graph();
            prefix_scan(graph.recorder(), counts, offsets, total, LateCount::fixed(K_CAPACITY))
                .value();

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

                graph.run().value();

                EXPECT_EQ(total[0], expected_total) << "step " << step;
                for (std::size_t i = 0; i <= K_CAPACITY; ++i)
                    ASSERT_EQ(offsets[i], expected_offsets[i])
                        << "step " << step << ", i = " << i;
            }

            EXPECT_EQ(graph.compile_count(), 1u);
        }

        /** @brief Tests that every node of one call shares one node_index on every step. */
        TEST(Medium_GraphPrimitivesPrefixScan, SharesOneNodeIndexAcrossStepsAndPasses)
        {
            // Two levels record a resolver, one up pass, two down passes and the closing write.
            constexpr std::size_t K_EXPECTED_PASSES = 5;
            constexpr std::size_t K_STEPS = 4;

            auto runtime = Runtime::create().value();
            auto counts = runtime.buffer<std::uint32_t>(K_TWO_LEVEL_CAPACITY).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_TWO_LEVEL_CAPACITY + 1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            for (std::size_t i = 0; i < K_TWO_LEVEL_CAPACITY; ++i)
                counts[i] = static_cast<std::uint32_t>((i % 7) + 1);

            auto graph = runtime.graph();
            graph.set_profiling(true);
            prefix_scan(graph.recorder(), counts, offsets, total,
                        LateCount::fixed(K_TWO_LEVEL_CAPACITY)).value();
            EXPECT_EQ(graph.recorder().node_count(), K_EXPECTED_PASSES);
            EXPECT_EQ(graph.recorder().rule_count(), 1u);

            const auto report = graph.run(K_STEPS).value();

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
        TEST(Medium_GraphPrimitivesPrefixScan, NodeWidthsMatchTheLevelPlan)
        {
            // The nodes are the resolver, up 0, the top down pass, down 0 and the closing write.
            const std::vector<std::uint32_t> expected_widths = {1u, 3u, 1u, 3u, 1u};

            auto runtime = Runtime::create().value();
            auto counts = runtime.buffer<std::uint32_t>(K_TWO_LEVEL_CAPACITY).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_TWO_LEVEL_CAPACITY + 1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            for (std::size_t i = 0; i < K_TWO_LEVEL_CAPACITY; ++i)
                counts[i] = static_cast<std::uint32_t>((i % 7) + 1);

            auto graph = runtime.graph();
            graph.set_profiling(true);
            prefix_scan(graph.recorder(), counts, offsets, total,
                        LateCount::fixed(K_TWO_LEVEL_CAPACITY)).value();

            const auto report = graph.run().value();

            ASSERT_EQ(report.node_events.size(), expected_widths.size());
            for (std::size_t i = 0; i < expected_widths.size(); ++i)
            {
                EXPECT_EQ(report.node_events[i].launched_width, expected_widths[i])
                    << "pass " << i;
                EXPECT_EQ(report.node_events[i].live_width, expected_widths[i]) << "pass " << i;
            }
        }

        /** @brief Tests that a capacity of zero writes a zero total and a zero offsets[0]. */
        TEST(Medium_GraphPrimitivesPrefixScan, ZeroCapacityWritesZeroTotal)
        {
            auto runtime = Runtime::create().value();
            auto counts = runtime.buffer<std::uint32_t>(1).value();
            auto offsets = runtime.buffer<std::uint32_t>(1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            offsets[0] = K_CANARY;
            total[0] = K_CANARY;

            auto graph = runtime.graph();
            prefix_scan(graph.recorder(), counts, offsets, total, LateCount::fixed(0)).value();
            graph.run().value();

            EXPECT_EQ(offsets[0], 0u);
            EXPECT_EQ(total[0], 0u);
        }

        /** @brief Tests that a short buffer is refused with invalid_graph and records nothing. */
        TEST(Medium_GraphPrimitivesPrefixScan, RejectsMisSizedArguments)
        {
            using Words = SushiRuntime::Buffer<std::uint32_t>;

            auto runtime = Runtime::create().value();
            auto counts = runtime.buffer<std::uint32_t>(8).value();
            auto offsets = runtime.buffer<std::uint32_t>(9).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            auto empty = runtime.buffer<std::uint32_t>(0).value();

            auto graph = runtime.graph();
            const auto refused_code = [&graph](const Words& c, const Words& o, const Words& t,
                                               std::size_t capacity)
            {
                const SushiRuntime::Result<void> recorded =
                    prefix_scan(graph.recorder(), c, o, t, LateCount::fixed(capacity));
                if (recorded)
                    return std::optional<SushiRuntime::ErrorCode>();

                return std::optional<SushiRuntime::ErrorCode>(recorded.error().code);
            };

            EXPECT_EQ(refused_code(counts, offsets, total, 9),
                      SushiRuntime::ErrorCode::InvalidGraph)
                << "nine counts do not fit a buffer of eight";
            EXPECT_EQ(refused_code(counts, counts, total, 8),
                      SushiRuntime::ErrorCode::InvalidGraph)
                << "eight offsets cannot hold the sentinel of eight counts";
            EXPECT_EQ(refused_code(counts, offsets, empty, 8),
                      SushiRuntime::ErrorCode::InvalidGraph)
                << "an empty total cannot receive the sum";
            EXPECT_EQ(graph.recorder().node_count(), 0u);
        }

        /** @brief Tests that a live count above its capacity fails the run. */
        TEST(Medium_GraphPrimitivesPrefixScan, LateBoundAboveCapacityFailsTheRun)
        {
            auto runtime = Runtime::create().value();
            auto counts = runtime.buffer<std::uint32_t>(100).value();
            auto offsets = runtime.buffer<std::uint32_t>(101).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();

            auto graph = runtime.graph();
            prefix_scan(graph.recorder(), counts, offsets, total,
                        LateCount::from_host(100, []() { return std::size_t{101}; })).value();
            EXPECT_FALSE(graph.run().has_value());
        }

        /** @brief Tests prefix_scan() with a device count on both layers. */
        TEST(Medium_GraphPrimitivesDeviceSized, PrefixScanChainMatchesScheduler)
        {
            using namespace DeviceSized;

            constexpr std::size_t K_CAPACITY = 70000;
            expect_chain_matches_scheduler([](Runtime& runtime)
            {
                Harness harness(runtime, K_CAPACITY);
                auto counts = runtime.buffer<std::uint32_t>(K_CAPACITY).value();
                auto offsets = runtime.buffer<std::uint32_t>(K_CAPACITY + 1).value();
                auto total = runtime.buffer<std::uint32_t>(1).value();
                for (std::size_t i = 0; i < K_CAPACITY; ++i)
                    counts[i] = static_cast<std::uint32_t>((i * 31u) % 9u);
                for (std::size_t i = 0; i <= K_CAPACITY; ++i)
                    offsets[i] = 0;

                auto graph = runtime.graph();
                harness.add_producer(graph);
                GraphPrimitives::prefix_scan(graph.recorder(), counts, offsets, total,
                                             harness.device_count(K_CAPACITY)).value();
                harness.add_live_digest(graph, offsets, 1);
                return harness.record(graph);
            });
        }

        /** @brief Tests that a device count above capacity fails a chained run of a primitive. */
        TEST(Medium_GraphPrimitivesDeviceSized, DeviceCountOverCapacityFailsAChainedRun)
        {
            using namespace DeviceSized;

            constexpr std::size_t K_CAPACITY = 100;

            auto runtime = Runtime::create().value();
            auto counter = runtime.buffer<std::uint32_t>(
                1, SushiRuntime::DeviceIndex{0}, SushiRuntime::Residency::Host).value();
            auto counts = runtime.buffer<std::uint32_t>(K_CAPACITY).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_CAPACITY + 1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            std::uint32_t* c = counter.data();

            auto graph = runtime.graph();
            graph.add(std::size_t{1}, Reads{}, Writes(counter),
                      [c](std::size_t)
                      {
                          c[0] = static_cast<std::uint32_t>(K_CAPACITY) + 1u;
                      });
            GraphPrimitives::prefix_scan(graph.recorder(), counts, offsets, total,
                                         LateCount::from_device(K_CAPACITY, counter)).value();
            EXPECT_FALSE(graph.run(4).has_value());

            // A later graph with a count inside the capacity runs clean on the same runtime.
            auto valid = runtime.graph();
            valid.add(std::size_t{1}, Reads{}, Writes(counter), [c](std::size_t) { c[0] = 3u; });
            GraphPrimitives::prefix_scan(valid.recorder(), counts, offsets, total,
                                         LateCount::from_device(K_CAPACITY, counter)).value();
            const auto report = valid.run(2).value();
            EXPECT_TRUE(report.is_successful);
            EXPECT_EQ(report.layer, ExecutionLayer::Chain);
        }
        /** @brief Tests that a scan over windows equals one over compact buffers. */
        TEST(Medium_GraphPrimitivesPrefixScan, WindowMatchesACompactBufferExactly)
        {
            constexpr std::size_t K_CAPACITY = K_TWO_LEVEL_CAPACITY;
            const GraphPrimitives::PrefixScanWindow window{9, 6, 2};
            const std::vector<std::uint32_t> host = cycling_counts(K_CAPACITY);

            auto runtime = Runtime::create().value();
            auto wide_counts = runtime.buffer<std::uint32_t>(window.counts_offset + K_CAPACITY + 4)
                                   .value();
            auto wide_offsets =
                runtime.buffer<std::uint32_t>(window.offsets_offset + K_CAPACITY + 1 + 4).value();
            auto wide_total = runtime.buffer<std::uint32_t>(window.total_offset + 2).value();
            auto counts = runtime.buffer<std::uint32_t>(K_CAPACITY).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_CAPACITY + 1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            for (std::size_t i = 0; i < wide_counts.size(); ++i)
                wide_counts[i] = 1000u;
            for (std::size_t i = 0; i < wide_offsets.size(); ++i)
                wide_offsets[i] = K_CANARY;
            for (std::size_t i = 0; i < wide_total.size(); ++i)
                wide_total[i] = K_CANARY;
            for (std::size_t i = 0; i < K_CAPACITY; ++i)
            {
                counts[i] = host[i];
                wide_counts[window.counts_offset + i] = host[i];
            }

            for (const std::size_t n : {std::size_t{0}, K_TILE + 1, K_CAPACITY})
            {
                auto graph = runtime.graph();
                prefix_scan(graph.recorder(), wide_counts, wide_offsets, wide_total,
                            LateCount::fixed(n), window).value();
                prefix_scan(graph.recorder(), counts, offsets, total, LateCount::fixed(n))
                    .value();
                graph.run().value();

                for (std::size_t i = 0; i <= n; ++i)
                    EXPECT_EQ(wide_offsets[window.offsets_offset + i], offsets[i])
                        << "n = " << n << ", offset " << i;
                EXPECT_EQ(wide_total[window.total_offset], total[0]) << "n = " << n;
                EXPECT_EQ(wide_offsets[window.offsets_offset - 1], K_CANARY) << "n = " << n;
                EXPECT_EQ(wide_total[window.total_offset - 1], K_CANARY) << "n = " << n;
            }
        }

        /** @brief Tests that a writer of an operand waits only when it hits the window. */
        TEST(Medium_GraphPrimitivesPrefixScan, WindowOrdersOnlyOverlappingWriters)
        {
            using WindowProbe::region_of;
            constexpr std::size_t K_CAPACITY = K_TWO_LEVEL_CAPACITY;
            const GraphPrimitives::PrefixScanWindow window{9, 6, 2};

            auto runtime = Runtime::create().value();
            auto counts = runtime.buffer<std::uint32_t>(window.counts_offset + K_CAPACITY + 4)
                              .value();
            auto offsets =
                runtime.buffer<std::uint32_t>(window.offsets_offset + K_CAPACITY + 1 + 4).value();
            auto total = runtime.buffer<std::uint32_t>(window.total_offset + 2).value();

            const auto predecessors = [&](const SushiRuntime::BufferRegion& written)
            {
                auto graph = runtime.graph();
                prefix_scan(graph.recorder(), counts, offsets, total,
                            LateCount::fixed(K_CAPACITY), window).value();
                return WindowProbe::predecessors_of_write(graph, written);
            };

            EXPECT_EQ(predecessors(region_of(counts, 0, window.counts_offset)), 0u);
            EXPECT_EQ(predecessors(region_of(counts, window.counts_offset + K_CAPACITY, 4)), 0u);
            EXPECT_GT(predecessors(region_of(counts, window.counts_offset + K_CAPACITY - 1, 1)),
                      0u);
            EXPECT_EQ(predecessors(region_of(offsets, window.offsets_offset + K_CAPACITY + 1, 4)),
                      0u);
            EXPECT_GT(predecessors(region_of(offsets, window.offsets_offset + K_CAPACITY, 1)), 0u);
            EXPECT_EQ(predecessors(region_of(total, 0, window.total_offset)), 0u);
            EXPECT_GT(predecessors(region_of(total, window.total_offset, 1)), 0u);
        }

        /** @brief Tests that a window past its buffer's end is refused, naming the operand. */
        TEST(Medium_GraphPrimitivesPrefixScan, RejectsAWindowPastItsBuffer)
        {
            auto runtime = Runtime::create().value();
            auto counts = runtime.buffer<std::uint32_t>(8).value();
            auto offsets = runtime.buffer<std::uint32_t>(9).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();

            const struct
            {
                GraphPrimitives::PrefixScanWindow window;
                const char* operand;
            } cases[] = {{{1, 0, 0}, "counts"}, {{0, 1, 0}, "offsets"}, {{0, 0, 1}, "total"}};

            auto graph = runtime.graph();
            for (const auto& refused : cases)
            {
                const SushiRuntime::Result<void> recorded = prefix_scan(
                    graph.recorder(), counts, offsets, total, LateCount::fixed(8), refused.window);
                ASSERT_FALSE(recorded.has_value()) << refused.operand;
                EXPECT_EQ(recorded.error().code, SushiRuntime::ErrorCode::InvalidGraph);
                EXPECT_NE(recorded.error().message.find(refused.operand), std::string::npos)
                    << recorded.error().message;
            }
            EXPECT_EQ(graph.recorder().node_count(), 0u) << "a refused call must record nothing";
        }

        /** @brief Tests that a named call names its nodes by stage and states their class. */
        TEST(Medium_GraphPrimitivesPrefixScan, NamedCallNamesItsNodesAndStatesTheirClass)
        {
            const GraphPrimitives::PrimitiveOptions options{
                "offsets", SushiRuntime::Determinism::Bitwise};

            auto runtime = Runtime::create().value();
            auto counts = runtime.buffer<std::uint32_t>(K_TWO_LEVEL_CAPACITY).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_TWO_LEVEL_CAPACITY + 1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();

            auto graph = runtime.graph();
            prefix_scan(graph.recorder(), counts, offsets, total,
                        LateCount::fixed(K_TWO_LEVEL_CAPACITY),
                        GraphPrimitives::PrefixScanWindow{}, options).value();

            EXPECT_EQ(PlanLabels::names_of(graph),
                      PlanLabels::staged("offsets", {"resolve", "up_sweep", "down_sweep",
                                                     "down_sweep", "close"}));
            for (const auto determinism : PlanLabels::determinisms_of(graph))
                EXPECT_EQ(determinism, SushiRuntime::Determinism::Bitwise);
        }

        /** @brief Tests that a call with no options keeps its nodes unnamed and Unstated. */
        TEST(Medium_GraphPrimitivesPrefixScan, UnnamedCallKeepsItsNodesUnnamed)
        {
            auto runtime = Runtime::create().value();
            auto counts = runtime.buffer<std::uint32_t>(K_TWO_LEVEL_CAPACITY).value();
            auto offsets = runtime.buffer<std::uint32_t>(K_TWO_LEVEL_CAPACITY + 1).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            const std::string unnamed = PlanLabels::unnamed_name(runtime);

            auto graph = runtime.graph();
            prefix_scan(graph.recorder(), counts, offsets, total,
                        LateCount::fixed(K_TWO_LEVEL_CAPACITY)).value();

            EXPECT_EQ(PlanLabels::names_of(graph), std::vector<std::string>(5, unnamed));
            for (const auto determinism : PlanLabels::determinisms_of(graph))
                EXPECT_EQ(determinism, SushiRuntime::Determinism::Unstated);
        }

        /** @brief Tests that options with no name are refused, naming the operand 'options'. */
        TEST(Medium_GraphPrimitivesPrefixScan, RejectsOptionsThatNameNothing)
        {
            auto runtime = Runtime::create().value();
            auto counts = runtime.buffer<std::uint32_t>(8).value();
            auto offsets = runtime.buffer<std::uint32_t>(9).value();
            auto total = runtime.buffer<std::uint32_t>(1).value();
            const GraphPrimitives::PrimitiveOptions unnamed{};
            const GraphPrimitives::PrimitiveOptions null_name{std::string_view{},
                                                              SushiRuntime::Determinism::Bitwise};
            const GraphPrimitives::PrimitiveOptions bad_class{
                "offsets", static_cast<SushiRuntime::Determinism>(250)};

            auto graph = runtime.graph();
            for (const auto& options : {unnamed, null_name, bad_class})
            {
                const SushiRuntime::Result<void> recorded = prefix_scan(
                    graph.recorder(), counts, offsets, total, LateCount::fixed(8),
                    GraphPrimitives::PrefixScanWindow{}, options);
                ASSERT_FALSE(recorded.has_value()) << "the call recorded instead of refusing";
                EXPECT_EQ(recorded.error().code, SushiRuntime::ErrorCode::InvalidGraph);
                EXPECT_NE(recorded.error().message.find("'options'"), std::string::npos)
                    << recorded.error().message;
            }
            EXPECT_EQ(graph.recorder().node_count(), 0u) << "a refused call must record nothing";
        }

    } // namespace Test
} // namespace SushiBLAS
