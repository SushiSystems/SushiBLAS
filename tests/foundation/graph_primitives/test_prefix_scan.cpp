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
 * @file test_prefix_scan.cpp
 * @brief Tests prefix_scan() against a host scan, across live counts, replays and node widths.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <ostream>
#include <set>
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
#include <SushiBLAS/test_runner/run_observer_capture.hpp>
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

            /// @brief Names the buffer type every operand of a scan has.
            using Words = SushiRuntime::Buffer<std::uint32_t>;

            /// @brief Holds the tile size the level plan of a scan is built on.
            constexpr std::size_t K_TILE = GraphPrimitives::K_REDUCE_TILE;

            /// @brief Holds a value no scan of these tests produces.
            constexpr std::uint32_t K_CANARY = 0xDEADBEEFu;

            /// @brief Holds a capacity of two levels: three tiles under one top tile.
            constexpr std::size_t K_TWO_LEVEL_CAPACITY = 2 * K_TILE + 88;

            /// @brief Holds the nodes a two-level call records: resolver, up, two downs, close.
            constexpr std::size_t K_TWO_LEVEL_PASSES = 5;

            /// @brief Holds the capacity of the scan whose live count changes between runs.
            constexpr std::size_t K_LATE_CAPACITY = 1000003;

            /// @brief Holds the capacity of the scans that read their count from the device.
            constexpr std::size_t K_DEVICE_CAPACITY = 100;

            /// @brief Places each operand of the window tests a few elements into its buffer.
            constexpr GraphPrimitives::PrefixScanWindow K_WINDOW{9, 6, 2};

            /// @brief Holds the three operands of one prefix_scan() call.
            struct ScanBuffers
            {
                /// @brief Holds the counts the scan reads.
                Words counts;
                /// @brief Holds the offsets the scan writes, the sum last.
                Words offsets;
                /// @brief Holds the sum the scan writes.
                Words total;
            };

            /// @brief Holds the sum and the sentinel offsets[n] one run published.
            struct ScanEnd
            {
                /// @brief Holds total[0] after the run.
                std::uint32_t total;
                /// @brief Holds offsets[n] after the run, n being the live count of the run.
                std::uint32_t sentinel;
            };

            /** @brief Returns whether two runs published the same sum and sentinel. */
            bool operator==(const ScanEnd& left, const ScanEnd& right)
            {
                return left.total == right.total && left.sentinel == right.sentinel;
            }

            /** @brief Prints a run's sum and sentinel in a failure message. */
            void PrintTo(const ScanEnd& end, std::ostream* out)
            {
                *out << "{total " << end.total << ", sentinel " << end.sentinel << "}";
            }

            /// @brief Names a short operand and the live count that does not fit it.
            struct MisSizedRow
            {
                /// @brief Names the row in the test report.
                const char* name;
                /// @brief Holds the element count of the counts buffer.
                std::size_t counts;
                /// @brief Holds the element count of the offsets buffer.
                std::size_t offsets;
                /// @brief Holds the element count of the total buffer.
                std::size_t total;
                /// @brief Holds the capacity the call is recorded with.
                std::size_t capacity;
            };

            /// @brief Names an operand and a window that reaches one element past its buffer.
            struct WindowRefusalRow
            {
                /// @brief Names the row in the test report, and the operand the error names.
                const char* name;
                /// @brief Holds the window that does not fit.
                GraphPrimitives::PrefixScanWindow window;
            };

            /// @brief Names a set of options that names no node or states no class.
            struct OptionRefusalRow
            {
                /// @brief Names the row in the test report.
                const char* name;
                /// @brief Holds the options the call is refused for.
                GraphPrimitives::PrimitiveOptions options;
            };

            /// @brief Names a pass of a two-level scan and the width it launches at.
            struct NodeWidthRow
            {
                /// @brief Names the row in the test report.
                const char* name;
                /// @brief Holds the place of the pass in the recorded order.
                std::size_t pass;
                /// @brief Holds the width the level plan gives the pass.
                std::uint32_t width;
            };

            /// @brief Holds what a scan over windows and the same scan over compact buffers wrote.
            struct WindowedScan
            {
                /// @brief Holds the offsets inside the window, the sum last.
                std::vector<std::uint32_t> window_offsets;
                /// @brief Holds the offsets of the compact scan, the sum last.
                std::vector<std::uint32_t> compact_offsets;
                /// @brief Holds the sum written inside the total window.
                std::uint32_t window_total;
                /// @brief Holds the sum of the compact scan.
                std::uint32_t compact_total;
                /// @brief Holds the offsets element just before the window.
                std::uint32_t offset_before_window;
                /// @brief Holds the total element just before the window.
                std::uint32_t total_before_window;
            };

            /** @brief Names a test case after its row. */
            template <typename Row>
            std::string row_name(const ::testing::TestParamInfo<Row>& info)
            {
                return info.param.name;
            }

            /** @brief Returns the n values generate(0) to generate(n - 1). */
            template <typename Generator>
            std::vector<std::uint32_t> counts_of(std::size_t n, Generator generate)
            {
                std::vector<std::uint32_t> counts(n);
                for (std::size_t i = 0; i < n; ++i)
                    counts[i] = static_cast<std::uint32_t>(generate(i));
                return counts;
            }

            /** @brief Returns n counts that cycle through 1 to 13. */
            std::vector<std::uint32_t> cycling_counts(std::size_t n)
            {
                return counts_of(n, [](std::size_t i) { return (i % 13) + 1; });
            }

            /** @brief Returns the counts 1 to n. */
            std::vector<std::uint32_t> ascending_counts(std::size_t n)
            {
                return counts_of(n, [](std::size_t i) { return i + 1; });
            }

            /** @brief Returns the counts of one replay step: (i + step) mod 5. */
            std::vector<std::uint32_t> step_counts(std::size_t n, std::size_t step)
            {
                return counts_of(n, [step](std::size_t i) { return (i + step) % 5; });
            }

            /**
             * @brief Returns the exclusive prefix sums of counts, modulo 2^32.
             * @return counts.size() + 1 offsets; the last one is the sum of every count.
             */
            std::vector<std::uint32_t> host_offsets(const std::vector<std::uint32_t>& counts)
            {
                std::vector<std::uint32_t> offsets(counts.size() + 1, 0);
                std::uint32_t running = 0;
                for (std::size_t i = 0; i < counts.size(); ++i)
                {
                    offsets[i] = running;
                    running += counts[i];
                }
                offsets[counts.size()] = running;
                return offsets;
            }

            /** @brief Writes values into buffer from element offset on. */
            void copy_into(const Words& buffer, const std::vector<std::uint32_t>& values,
                           std::size_t offset = 0)
            {
                for (std::size_t i = 0; i < values.size(); ++i)
                    buffer[offset + i] = values[i];
            }

            /** @brief Sets every element of buffer to value. */
            void fill_with(const Words& buffer, std::uint32_t value)
            {
                for (std::size_t i = 0; i < buffer.size(); ++i)
                    buffer[i] = value;
            }

            /** @brief Returns elements [begin, begin + count) of buffer. */
            std::vector<std::uint32_t> words_of(const Words& buffer, std::size_t begin,
                                                std::size_t count)
            {
                std::vector<std::uint32_t> words(count);
                for (std::size_t i = 0; i < count; ++i)
                    words[i] = buffer[begin + i];
                return words;
            }

            /** @brief Returns counts, offsets and total buffers of the given element counts. */
            ScanBuffers scan_buffers(Runtime& runtime, std::size_t counts, std::size_t offsets,
                                     std::size_t total)
            {
                return {runtime.buffer<std::uint32_t>(counts).value(),
                        runtime.buffer<std::uint32_t>(offsets).value(),
                        runtime.buffer<std::uint32_t>(total).value()};
            }

            /** @brief Returns compact buffers for a scan of the given capacity. */
            ScanBuffers buffers_for(Runtime& runtime, std::size_t capacity)
            {
                return scan_buffers(runtime, capacity, capacity + 1, 1);
            }

            /** @brief Returns compact buffers of two levels, counts cycling through 1 to 7. */
            ScanBuffers two_level_buffers(Runtime& runtime)
            {
                ScanBuffers buffers = buffers_for(runtime, K_TWO_LEVEL_CAPACITY);
                copy_into(buffers.counts, counts_of(K_TWO_LEVEL_CAPACITY,
                                                    [](std::size_t i) { return (i % 7) + 1; }));
                return buffers;
            }

            /** @brief Records a two-level scan over buffers into graph, with options if any. */
            void record_two_level(SushiRuntime::Graph& graph, const ScanBuffers& buffers,
                                  const std::optional<GraphPrimitives::PrimitiveOptions>& options)
            {
                prefix_scan(graph.recorder(), buffers.counts, buffers.offsets, buffers.total,
                            LateCount::fixed(K_TWO_LEVEL_CAPACITY),
                            GraphPrimitives::PrefixScanWindow{}, options).value();
            }

            /** @brief Runs a two-level scan for steps steps and returns the node events seen. */
            std::vector<SushiRuntime::NodeEventRecord> two_level_events(Runtime& runtime,
                                                                        std::size_t steps)
            {
                const ScanBuffers buffers = two_level_buffers(runtime);
                auto graph = runtime.graph();
                const RunObserverCapture capture(runtime);
                record_two_level(graph, buffers, std::nullopt);
                graph.run(steps).value();
                return capture.records();
            }

            /** @brief Returns the distinct node_index values of events. */
            std::set<std::uint32_t>
            node_indices_of(const std::vector<SushiRuntime::NodeEventRecord>& events)
            {
                std::set<std::uint32_t> indices;
                for (const auto& record : events)
                    indices.insert(record.node_index);
                return indices;
            }

            /** @brief Returns how many events each invocation saw, keyed by invocation. */
            std::map<std::uint32_t, std::size_t>
            passes_per_invocation(const std::vector<SushiRuntime::NodeEventRecord>& events)
            {
                std::map<std::uint32_t, std::size_t> passes;
                for (const auto& record : events)
                    ++passes[record.invocation];
                return passes;
            }

            /**
             * @brief Writes each step's counts, runs graph once per step and returns its offsets.
             * @return Per step, the offsets 0 to n with offsets[n] replaced by total[0].
             */
            std::vector<std::vector<std::uint32_t>>
            replay_offsets(SushiRuntime::Graph& graph, const ScanBuffers& buffers,
                           const std::vector<std::vector<std::uint32_t>>& steps)
            {
                std::vector<std::vector<std::uint32_t>> results;
                for (const auto& counts : steps)
                {
                    copy_into(buffers.counts, counts);
                    graph.run().value();
                    std::vector<std::uint32_t> offsets =
                        words_of(buffers.offsets, 0, counts.size() + 1);
                    offsets.back() = buffers.total[0];
                    results.push_back(offsets);
                }
                return results;
            }

            /** @brief Returns the host scan of each step, as replay_offsets() returns it. */
            std::vector<std::vector<std::uint32_t>>
            host_offsets_of_each(const std::vector<std::vector<std::uint32_t>>& steps)
            {
                std::vector<std::vector<std::uint32_t>> results;
                for (const auto& counts : steps)
                    results.push_back(host_offsets(counts));
                return results;
            }

            /**
             * @brief Records a node that sets counter to live_count, then a scan sized by it.
             * @param counter Holds one host-resident element the scan reads its count from.
             */
            void record_device_counted_scan(SushiRuntime::Graph& graph, Words& counter,
                                            const ScanBuffers& buffers, std::uint32_t live_count)
            {
                std::uint32_t* c = counter.data();
                graph.add(std::size_t{1}, SushiRuntime::Reads{}, SushiRuntime::Writes(counter),
                          [c, live_count](std::size_t)
                          {
                              c[0] = live_count;
                          });
                prefix_scan(graph.recorder(), buffers.counts, buffers.offsets, buffers.total,
                            LateCount::from_device(K_DEVICE_CAPACITY, counter)).value();
            }

            /** @brief Returns a one-element host-resident counter for a device count. */
            Words device_counter(Runtime& runtime)
            {
                return runtime.buffer<std::uint32_t>(1, SushiRuntime::DeviceIndex{0},
                                                     SushiRuntime::Residency::Host).value();
            }

            /** @brief Scans K_WINDOW of wide buffers and compact buffers at live count n. */
            WindowedScan scan_window_and_compact(Runtime& runtime, std::size_t n)
            {
                constexpr std::size_t K_CAPACITY = K_TWO_LEVEL_CAPACITY;
                const std::vector<std::uint32_t> host = cycling_counts(K_CAPACITY);
                const ScanBuffers wide =
                    scan_buffers(runtime, K_WINDOW.counts_offset + K_CAPACITY + 4,
                                 K_WINDOW.offsets_offset + K_CAPACITY + 1 + 4,
                                 K_WINDOW.total_offset + 2);
                const ScanBuffers compact = buffers_for(runtime, K_CAPACITY);
                fill_with(wide.counts, 1000u);
                fill_with(wide.offsets, K_CANARY);
                fill_with(wide.total, K_CANARY);
                copy_into(wide.counts, host, K_WINDOW.counts_offset);
                copy_into(compact.counts, host);

                auto graph = runtime.graph();
                prefix_scan(graph.recorder(), wide.counts, wide.offsets, wide.total,
                            LateCount::fixed(n), K_WINDOW).value();
                prefix_scan(graph.recorder(), compact.counts, compact.offsets, compact.total,
                            LateCount::fixed(n)).value();
                graph.run().value();

                return {words_of(wide.offsets, K_WINDOW.offsets_offset, n + 1),
                        words_of(compact.offsets, 0, n + 1),
                        wide.total[K_WINDOW.total_offset],
                        compact.total[0],
                        wide.offsets[K_WINDOW.offsets_offset - 1],
                        wide.total[K_WINDOW.total_offset - 1]};
            }

            /** @brief Holds a scan whose live count a host provider reads between runs. */
            class LateBoundScan final
            {
                public:
                    /** @brief Records the scan over K_LATE_CAPACITY counts of one. */
                    LateBoundScan()
                        : runtime_(Runtime::create().value()),
                          buffers_(buffers_for(runtime_, K_LATE_CAPACITY)),
                          graph_(runtime_.graph())
                    {
                        fill_with(buffers_.counts, 1u);
                        prefix_scan(graph_.recorder(), buffers_.counts, buffers_.offsets,
                                    buffers_.total,
                                    LateCount::from_host(K_LATE_CAPACITY,
                                                         [this]() { return live_n_; })).value();
                    }

                    LateBoundScan(const LateBoundScan&) = delete;
                    LateBoundScan& operator=(const LateBoundScan&) = delete;
                    LateBoundScan(LateBoundScan&&) = delete;
                    LateBoundScan& operator=(LateBoundScan&&) = delete;
                    ~LateBoundScan() = default;

                    /** @brief Runs once per live count and returns what each run published. */
                    std::vector<ScanEnd> ends_for(const std::vector<std::size_t>& live_counts)
                    {
                        std::vector<ScanEnd> ends;
                        for (const std::size_t n : live_counts)
                        {
                            live_n_ = n;
                            graph_.run().value();
                            ends.push_back({buffers_.total[0], buffers_.offsets[n]});
                        }
                        return ends;
                    }

                    /** @brief Returns how many times the graph compiled. */
                    std::size_t compile_count()
                    {
                        return graph_.compile_count();
                    }

                private:
                    /// @brief Holds the runtime the scan runs on.
                    Runtime runtime_;
                    /// @brief Holds the operands of the scan.
                    ScanBuffers buffers_;
                    /// @brief Holds the live count the host provider returns.
                    std::size_t live_n_ = 0;
                    /// @brief Holds the graph the scan is recorded into.
                    SushiRuntime::Graph graph_;
            };
        } // namespace

        /// @brief Runs a scan at one live count on the tile and level edges.
        class Medium_GraphPrimitivesPrefixScanTileEdges
            : public ::testing::TestWithParam<std::size_t>
        {
        };

        /** @brief Tests exact agreement with the host scan at the tile and level edges. */
        TEST_P(Medium_GraphPrimitivesPrefixScanTileEdges, AgreesWithTheHostScanAtTileEdges)
        {
            const std::size_t n = GetParam();
            auto runtime = Runtime::create().value();
            const ScanBuffers buffers =
                scan_buffers(runtime, std::max<std::size_t>(n, 1), n + 2, 1);
            const std::vector<std::uint32_t> host = cycling_counts(n);
            copy_into(buffers.counts, host);
            fill_with(buffers.offsets, K_CANARY);
            fill_with(buffers.total, K_CANARY);
            const std::vector<std::uint32_t> expected = host_offsets(host);

            auto graph = runtime.graph();
            prefix_scan(graph.recorder(), buffers.counts, buffers.offsets, buffers.total,
                        LateCount::fixed(n)).value();
            graph.run().value();

            EXPECT_EQ(buffers.total[0], expected.back());
            EXPECT_EQ(words_of(buffers.offsets, 0, n + 1), expected);
            EXPECT_EQ(buffers.offsets[n + 1], K_CANARY)
                << "a write reached past the declared window";
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_GraphPrimitivesPrefixScanTileEdges,
                                 ::testing::Values(std::size_t{0}, std::size_t{1}, K_TILE - 1,
                                                   K_TILE, K_TILE + 1, K_TILE * K_TILE,
                                                   K_TILE * K_TILE + 1, std::size_t{1000000}));

        /** @brief Tests that offsets[n] equals the total the scan publishes. */
        TEST(Medium_GraphPrimitivesPrefixScan, OffsetsAtLiveCountEqualsTotal)
        {
            constexpr std::size_t K_COUNT = K_TILE + 1;

            auto runtime = Runtime::create().value();
            const ScanBuffers buffers = buffers_for(runtime, K_COUNT);
            copy_into(buffers.counts, ascending_counts(K_COUNT));

            auto graph = runtime.graph();
            prefix_scan(graph.recorder(), buffers.counts, buffers.offsets, buffers.total,
                        LateCount::fixed(K_COUNT)).value();
            graph.run().value();

            EXPECT_EQ(buffers.total[0], static_cast<std::uint32_t>(K_COUNT * (K_COUNT + 1) / 2));
            EXPECT_EQ(buffers.offsets[K_COUNT], buffers.total[0]);
        }

        /** @brief Tests that each run of a late-bound scan sums the live count of that run. */
        TEST(Medium_GraphPrimitivesPrefixScan, LateBoundChangingCountScansEachLiveCount)
        {
            LateBoundScan scan;

            EXPECT_EQ(scan.ends_for({1000, 700000, 3}),
                      (std::vector<ScanEnd>{{1000u, 1000u}, {700000u, 700000u}, {3u, 3u}}));
        }

        /** @brief Tests that a live count that changes between runs keeps one compile. */
        TEST(Medium_GraphPrimitivesPrefixScan, LateBoundChangingCountKeepsOneCompile)
        {
            LateBoundScan scan;
            (void)scan.ends_for({1000, 700000, 3});

            EXPECT_EQ(scan.compile_count(), 1u)
                << "a live count change between runs must not trigger a recompile";
        }

        /** @brief Tests that each run scans the counts the buffer holds when it runs. */
        TEST(Medium_GraphPrimitivesPrefixScan, EachRunScansItsOwnCounts)
        {
            constexpr std::size_t K_CAPACITY = 300;
            const std::vector<std::vector<std::uint32_t>> steps = {
                step_counts(K_CAPACITY, 0), step_counts(K_CAPACITY, 1),
                step_counts(K_CAPACITY, 2), step_counts(K_CAPACITY, 3)};

            auto runtime = Runtime::create().value();
            const ScanBuffers buffers = buffers_for(runtime, K_CAPACITY);
            auto graph = runtime.graph();
            prefix_scan(graph.recorder(), buffers.counts, buffers.offsets, buffers.total,
                        LateCount::fixed(K_CAPACITY)).value();

            EXPECT_EQ(replay_offsets(graph, buffers, steps), host_offsets_of_each(steps));
            EXPECT_EQ(graph.compile_count(), 1u);
        }

        /** @brief Tests that a two-level call records five passes under one rule. */
        TEST(Medium_GraphPrimitivesPrefixScan, TwoLevelCallRecordsFivePassesUnderOneRule)
        {
            auto runtime = Runtime::create().value();
            const ScanBuffers buffers = two_level_buffers(runtime);
            auto graph = runtime.graph();
            record_two_level(graph, buffers, std::nullopt);

            EXPECT_EQ(graph.recorder().node_count(), K_TWO_LEVEL_PASSES);
            EXPECT_EQ(graph.recorder().rule_count(), 1u);
        }

        /** @brief Tests that every node of one call shares one node_index on every step. */
        TEST(Medium_GraphPrimitivesPrefixScan, SharesOneNodeIndexAcrossStepsAndPasses)
        {
            auto runtime = Runtime::create().value();

            EXPECT_EQ(node_indices_of(two_level_events(runtime, 4)).size(), 1u)
                << "every pass of one prefix_scan() call must share one node_index";
        }

        /** @brief Tests that every step of a two-level scan runs all five passes. */
        TEST(Medium_GraphPrimitivesPrefixScan, EveryStepRunsEveryPass)
        {
            auto runtime = Runtime::create().value();

            EXPECT_EQ(passes_per_invocation(two_level_events(runtime, 4)),
                      (std::map<std::uint32_t, std::size_t>{{0u, K_TWO_LEVEL_PASSES},
                                                            {1u, K_TWO_LEVEL_PASSES},
                                                            {2u, K_TWO_LEVEL_PASSES},
                                                            {3u, K_TWO_LEVEL_PASSES}}));
        }

        /// @brief Runs one pass of a two-level scan against its width in the level plan.
        class Medium_GraphPrimitivesPrefixScanNodeWidth
            : public ::testing::TestWithParam<NodeWidthRow>
        {
        };

        /** @brief Tests that each node launches at the width its pass has in the level plan. */
        TEST_P(Medium_GraphPrimitivesPrefixScanNodeWidth, NodeWidthsMatchTheLevelPlan)
        {
            const NodeWidthRow& row = GetParam();
            auto runtime = Runtime::create().value();
            const auto events = two_level_events(runtime, 1);

            ASSERT_EQ(events.size(), K_TWO_LEVEL_PASSES);
            EXPECT_EQ(events[row.pass].launched_width, row.width);
            EXPECT_EQ(events[row.pass].live_width, row.width);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_GraphPrimitivesPrefixScanNodeWidth,
                                 ::testing::Values(NodeWidthRow{"Resolver", 0, 1u},
                                                   NodeWidthRow{"UpLevel0", 1, 3u},
                                                   NodeWidthRow{"DownTop", 2, 1u},
                                                   NodeWidthRow{"DownLevel0", 3, 3u},
                                                   NodeWidthRow{"Close", 4, 1u}),
                                 row_name<NodeWidthRow>);

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

        /// @brief Records a scan whose buffers are too short for its capacity.
        class Medium_GraphPrimitivesPrefixScanMisSized
            : public ::testing::TestWithParam<MisSizedRow>
        {
        };

        /** @brief Tests that a short buffer is refused with invalid_graph and records nothing. */
        TEST_P(Medium_GraphPrimitivesPrefixScanMisSized, RejectsMisSizedArguments)
        {
            const MisSizedRow& row = GetParam();
            auto runtime = Runtime::create().value();
            const ScanBuffers buffers = scan_buffers(runtime, row.counts, row.offsets, row.total);
            auto graph = runtime.graph();

            const SushiRuntime::Result<void> recorded =
                prefix_scan(graph.recorder(), buffers.counts, buffers.offsets, buffers.total,
                            LateCount::fixed(row.capacity));

            ASSERT_FALSE(recorded.has_value()) << "the call recorded instead of refusing";
            EXPECT_EQ(recorded.error().code, SushiRuntime::ErrorCode::InvalidGraph);
            EXPECT_EQ(graph.recorder().node_count(), 0u) << "a refused call must record nothing";
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_GraphPrimitivesPrefixScanMisSized,
            ::testing::Values(MisSizedRow{"NineCountsInEight", 8, 9, 1, 9},
                              MisSizedRow{"NoRoomForTheSentinel", 8, 8, 1, 8},
                              MisSizedRow{"EmptyTotal", 8, 9, 0, 8}),
            row_name<MisSizedRow>);

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
                const ScanBuffers buffers = buffers_for(runtime, K_CAPACITY);
                copy_into(buffers.counts,
                          counts_of(K_CAPACITY, [](std::size_t i) { return (i * 31u) % 9u; }));
                fill_with(buffers.offsets, 0u);

                auto graph = runtime.graph();
                harness.add_producer(graph);
                GraphPrimitives::prefix_scan(graph.recorder(), buffers.counts, buffers.offsets,
                                             buffers.total,
                                             harness.device_count(K_CAPACITY)).value();
                harness.add_live_digest(graph, buffers.offsets, 1);
                return harness.record(graph);
            });
        }

        /** @brief Tests that a device count above capacity fails a chained run of a primitive. */
        TEST(Medium_GraphPrimitivesDeviceSized, DeviceCountOverCapacityFailsAChainedRun)
        {
            auto runtime = Runtime::create().value();
            auto counter = device_counter(runtime);
            const ScanBuffers buffers = buffers_for(runtime, K_DEVICE_CAPACITY);

            auto graph = runtime.graph();
            record_device_counted_scan(graph, counter, buffers,
                                       static_cast<std::uint32_t>(K_DEVICE_CAPACITY) + 1u);

            EXPECT_FALSE(graph.run(4).has_value());
        }

        /** @brief Tests that a chained run after one that failed on its count runs clean. */
        TEST(Medium_GraphPrimitivesDeviceSized, ChainedRunAfterACountFailureRunsClean)
        {
            using namespace DeviceSized;

            auto runtime = Runtime::create().value();
            auto counter = device_counter(runtime);
            const ScanBuffers buffers = buffers_for(runtime, K_DEVICE_CAPACITY);
            auto failed = runtime.graph();
            record_device_counted_scan(failed, counter, buffers,
                                       static_cast<std::uint32_t>(K_DEVICE_CAPACITY) + 1u);
            ASSERT_FALSE(failed.run(4).has_value());

            auto valid = runtime.graph();
            record_device_counted_scan(valid, counter, buffers, 3u);
            const auto report = valid.run(2).value();

            EXPECT_TRUE(report.is_successful);
            EXPECT_EQ(report.layer, ExecutionLayer::Chain);
        }

        /// @brief Scans a window and compact buffers at one live count.
        class Medium_GraphPrimitivesPrefixScanWindow
            : public ::testing::TestWithParam<std::size_t>
        {
        };

        /** @brief Tests that a scan over windows equals one over compact buffers. */
        TEST_P(Medium_GraphPrimitivesPrefixScanWindow, WindowMatchesACompactBufferExactly)
        {
            auto runtime = Runtime::create().value();
            const WindowedScan scan = scan_window_and_compact(runtime, GetParam());

            EXPECT_EQ(scan.window_offsets, scan.compact_offsets);
            EXPECT_EQ(scan.window_total, scan.compact_total);
        }

        /** @brief Tests that a windowed scan leaves the element before each output window. */
        TEST_P(Medium_GraphPrimitivesPrefixScanWindow, WindowLeavesTheElementBeforeItUntouched)
        {
            auto runtime = Runtime::create().value();
            const WindowedScan scan = scan_window_and_compact(runtime, GetParam());

            EXPECT_EQ(scan.offset_before_window, K_CANARY);
            EXPECT_EQ(scan.total_before_window, K_CANARY);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_GraphPrimitivesPrefixScanWindow,
                                 ::testing::Values(std::size_t{0}, K_TILE + 1,
                                                   K_TWO_LEVEL_CAPACITY));

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

        /// @brief Records a scan whose window reaches past one operand's buffer.
        class Medium_GraphPrimitivesPrefixScanWindowRefusal
            : public ::testing::TestWithParam<WindowRefusalRow>
        {
        };

        /** @brief Tests that a window past its buffer's end is refused, naming the operand. */
        TEST_P(Medium_GraphPrimitivesPrefixScanWindowRefusal, RejectsAWindowPastItsBuffer)
        {
            const WindowRefusalRow& row = GetParam();
            auto runtime = Runtime::create().value();
            const ScanBuffers buffers = buffers_for(runtime, 8);
            auto graph = runtime.graph();

            const SushiRuntime::Result<void> recorded =
                prefix_scan(graph.recorder(), buffers.counts, buffers.offsets, buffers.total,
                            LateCount::fixed(8), row.window);

            ASSERT_FALSE(recorded.has_value()) << "the call recorded instead of refusing";
            EXPECT_EQ(recorded.error().code, SushiRuntime::ErrorCode::InvalidGraph);
            EXPECT_NE(recorded.error().message.find(row.name), std::string::npos)
                << recorded.error().message;
            EXPECT_EQ(graph.recorder().node_count(), 0u) << "a refused call must record nothing";
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_GraphPrimitivesPrefixScanWindowRefusal,
            ::testing::Values(WindowRefusalRow{"counts", {1, 0, 0}},
                              WindowRefusalRow{"offsets", {0, 1, 0}},
                              WindowRefusalRow{"total", {0, 0, 1}}),
            row_name<WindowRefusalRow>);

        /** @brief Tests that a named call names its nodes by stage. */
        TEST(Medium_GraphPrimitivesPrefixScan, NamedCallNamesItsNodesByStage)
        {
            auto runtime = Runtime::create().value();
            const ScanBuffers buffers = two_level_buffers(runtime);
            auto graph = runtime.graph();
            record_two_level(graph, buffers,
                             GraphPrimitives::PrimitiveOptions{
                                 "offsets", SushiRuntime::Determinism::Bitwise});

            EXPECT_EQ(PlanLabels::names_of(graph),
                      PlanLabels::staged("offsets", {"resolve", "up_sweep", "down_sweep",
                                                     "down_sweep", "close"}));
        }

        /** @brief Tests that a named call states its class on every node. */
        TEST(Medium_GraphPrimitivesPrefixScan, NamedCallStatesTheClassOfEveryNode)
        {
            auto runtime = Runtime::create().value();
            const ScanBuffers buffers = two_level_buffers(runtime);
            auto graph = runtime.graph();
            record_two_level(graph, buffers,
                             GraphPrimitives::PrimitiveOptions{
                                 "offsets", SushiRuntime::Determinism::Bitwise});

            EXPECT_EQ(PlanLabels::determinisms_of(graph),
                      std::vector<SushiRuntime::Determinism>(
                          K_TWO_LEVEL_PASSES, SushiRuntime::Determinism::Bitwise));
        }

        /** @brief Tests that a call with no options keeps its nodes unnamed. */
        TEST(Medium_GraphPrimitivesPrefixScan, UnnamedCallKeepsItsNodesUnnamed)
        {
            auto runtime = Runtime::create().value();
            const ScanBuffers buffers = two_level_buffers(runtime);
            const std::string unnamed = PlanLabels::unnamed_name(runtime);
            auto graph = runtime.graph();
            record_two_level(graph, buffers, std::nullopt);

            EXPECT_EQ(PlanLabels::names_of(graph),
                      std::vector<std::string>(K_TWO_LEVEL_PASSES, unnamed));
        }

        /** @brief Tests that a call with no options leaves every node Unstated. */
        TEST(Medium_GraphPrimitivesPrefixScan, UnnamedCallLeavesEveryNodeUnstated)
        {
            auto runtime = Runtime::create().value();
            const ScanBuffers buffers = two_level_buffers(runtime);
            auto graph = runtime.graph();
            record_two_level(graph, buffers, std::nullopt);

            EXPECT_EQ(PlanLabels::determinisms_of(graph),
                      std::vector<SushiRuntime::Determinism>(
                          K_TWO_LEVEL_PASSES, SushiRuntime::Determinism::Unstated));
        }

        /// @brief Records a scan with options that name nothing or state no class.
        class Medium_GraphPrimitivesPrefixScanOptionRefusal
            : public ::testing::TestWithParam<OptionRefusalRow>
        {
        };

        /** @brief Tests that options with no name are refused, naming the operand 'options'. */
        TEST_P(Medium_GraphPrimitivesPrefixScanOptionRefusal, RejectsOptionsThatNameNothing)
        {
            auto runtime = Runtime::create().value();
            const ScanBuffers buffers = buffers_for(runtime, 8);
            auto graph = runtime.graph();

            const SushiRuntime::Result<void> recorded =
                prefix_scan(graph.recorder(), buffers.counts, buffers.offsets, buffers.total,
                            LateCount::fixed(8), GraphPrimitives::PrefixScanWindow{},
                            GetParam().options);

            ASSERT_FALSE(recorded.has_value()) << "the call recorded instead of refusing";
            EXPECT_EQ(recorded.error().code, SushiRuntime::ErrorCode::InvalidGraph);
            EXPECT_NE(recorded.error().message.find("'options'"), std::string::npos)
                << recorded.error().message;
            EXPECT_EQ(graph.recorder().node_count(), 0u) << "a refused call must record nothing";
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_GraphPrimitivesPrefixScanOptionRefusal,
            ::testing::Values(
                OptionRefusalRow{"Unnamed", GraphPrimitives::PrimitiveOptions{}},
                OptionRefusalRow{"NullName",
                                 GraphPrimitives::PrimitiveOptions{
                                     std::string_view{}, SushiRuntime::Determinism::Bitwise}},
                OptionRefusalRow{"UnknownClass",
                                 GraphPrimitives::PrimitiveOptions{
                                     "offsets", static_cast<SushiRuntime::Determinism>(250)}}),
            row_name<OptionRefusalRow>);

    } // namespace Test
} // namespace SushiBLAS
