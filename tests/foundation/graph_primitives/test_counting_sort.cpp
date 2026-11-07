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
 * @file test_counting_sort.cpp
 * @brief Tests counting_sort(): stability, digit boundaries, late-bound replay and node widths.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/graph/dynamic_graph.hpp>
#include <SushiRuntime/api/graph/graph.hpp>
#include <SushiRuntime/api/runtime/runtime.hpp>
#include <SushiRuntime/api/vocabulary/dependencies.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>
#include <SushiRuntime/core/determinism.hpp>
#include <SushiRuntime/core/error.hpp>
#include <SushiRuntime/report/node_event_record.hpp>

#include <SushiBLAS/graph_primitives/counting_sort.hpp>
#include <SushiBLAS/graph_primitives/host_primitive_recorder.hpp>
#include <SushiBLAS/test_runner/run_observer_capture.hpp>
#include <SushiBLAS/graph_primitives/prefix_scan.hpp>
#include <SushiBLAS/graph_primitives/primitive_options.hpp>
#include <SushiBLAS/graph_primitives/radix_plan.hpp>
#include "device_sized_harness.hpp"
#include "host_nodes.hpp"
#include "plan_labels.hpp"
#include "window_probe.hpp"
#include "runtime_port_calls.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using RuntimePort::counting_sort;
            using RuntimePort::prefix_scan;
            using SushiRuntime::Determinism;
            using SushiRuntime::LateCount;
            using SushiRuntime::NodeEventRecord;
            using SushiRuntime::Runtime;
            using Words = SushiRuntime::Buffer<std::uint32_t>;
            using Window = GraphPrimitives::CountingSortWindow;

            /// @brief Holds a value no sort of these tests writes.
            constexpr std::uint32_t K_CANARY = 0xDEADBEEFu;

            /// @brief Holds the nodes one radix digit records.
            constexpr std::size_t K_NODES_PER_DIGIT = 4;

            /// @brief Holds the element count of the seeded duplicate keys.
            constexpr std::size_t K_DUPLICATE_COUNT = 4000;

            /// @brief Holds the key range of the seeded duplicate keys.
            constexpr std::size_t K_DUPLICATE_KEY_RANGE = 32;

            /// @brief Holds the element count of the sorts whose dispatches are captured.
            constexpr std::size_t K_CAPTURED_COUNT = 300;

            /// @brief Holds the key range of the sorts whose dispatches are captured.
            constexpr std::size_t K_CAPTURED_KEY_RANGE = 50;

            /// @brief Holds the element count of the sort run once per step on new keys.
            constexpr std::size_t K_STEP_COUNT = 500;

            /// @brief Holds the key range of the sort run once per step on new keys.
            constexpr std::size_t K_STEP_KEY_RANGE = 20;

            /// @brief Holds the key range of the sorts whose node labels are read.
            constexpr std::size_t K_LABEL_KEY_RANGE = 4;

            /// @brief Holds the element count of the device-sized sort.
            constexpr std::size_t K_DEVICE_COUNT_CAPACITY = 1000;

            /// @brief Holds the key range capacity of the device-sized sort.
            constexpr std::size_t K_DEVICE_KEY_CAPACITY = std::size_t{1} << 17;

            /// @brief Holds the buckets the device-sized keys fall in.
            constexpr std::size_t K_DEVICE_BUCKETS = 32;

            /// @brief Holds the words the device-sized digest reads: indices, offsets and total.
            constexpr std::size_t K_DEVICE_PACKED = K_DEVICE_COUNT_CAPACITY + K_DEVICE_BUCKETS + 2;

            /// @brief Holds the window the window tests place every operand at.
            constexpr Window K_WINDOW{7, 5, 3, 1, 2};

            /// @brief Holds the spare elements past each window of the window tests.
            constexpr std::size_t K_WINDOW_TAIL = 3;

            /// @brief Holds the element count of the window probe sort.
            constexpr std::size_t K_PROBE_COUNT = 600;

            /// @brief Holds the key range of the window probe sort.
            constexpr std::size_t K_PROBE_KEY_RANGE = 300;

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

            /** @brief Returns the seeded keys of 32 buckets the stability tests sort. */
            std::vector<std::uint32_t> duplicate_keys()
            {
                return seeded_keys(K_DUPLICATE_COUNT, K_DUPLICATE_KEY_RANGE, 20260916);
            }

            /** @brief Returns 4000 seeded keys from 0x100 to 0x400, all of low byte 0. */
            std::vector<std::uint32_t> low_digit_tied_keys()
            {
                const std::array<std::uint32_t, 4> alphabet = {0x0100u, 0x0200u, 0x0300u, 0x0400u};
                std::mt19937 generator(20260917);
                std::uniform_int_distribution<std::size_t> pick(0, alphabet.size() - 1);
                std::vector<std::uint32_t> keys(K_DUPLICATE_COUNT);
                for (std::uint32_t& key : keys)
                    key = alphabet[pick(generator)];
                return keys;
            }

            /** @brief Returns the words 0, 1, ..., count - 1. */
            std::vector<std::uint32_t> ascending_words(std::size_t count)
            {
                std::vector<std::uint32_t> words(count);
                std::iota(words.begin(), words.end(), std::uint32_t{0});
                return words;
            }

            /** @brief Returns count keys that cycle through [0, key_range). */
            std::vector<std::uint32_t> cycled_keys(std::size_t count, std::size_t key_range)
            {
                std::vector<std::uint32_t> keys(count);
                for (std::size_t i = 0; i < count; ++i)
                    keys[i] = static_cast<std::uint32_t>(i % key_range);
                return keys;
            }

            /** @brief Returns the keys the repeated sort sorts on run step. */
            std::vector<std::uint32_t> step_keys(std::uint32_t step)
            {
                std::vector<std::uint32_t> keys(K_STEP_COUNT);
                for (std::size_t i = 0; i < K_STEP_COUNT; ++i)
                    keys[i] = static_cast<std::uint32_t>((i * 7 + step * 3) % K_STEP_KEY_RANGE);
                return keys;
            }

            /** @brief Returns the keys of the device-sized sort, spread over its 32 buckets. */
            std::vector<std::uint32_t> device_sized_keys()
            {
                std::vector<std::uint32_t> keys(K_DEVICE_COUNT_CAPACITY);
                for (std::size_t i = 0; i < K_DEVICE_COUNT_CAPACITY; ++i)
                    keys[i] = static_cast<std::uint32_t>((i * 7u + i / 13u) % K_DEVICE_BUCKETS);
                return keys;
            }

            /** @brief Returns the exclusive sums of values, then their total. */
            std::vector<std::uint32_t> exclusive_sums(const std::vector<std::uint32_t>& values)
            {
                std::vector<std::uint32_t> sums(values.size() + 1, 0);
                std::partial_sum(values.begin(), values.end(), sums.begin() + 1);
                return sums;
            }

            /** @brief Returns a buffer of size elements, each holding value. */
            Words filled(Runtime& runtime, std::size_t size, std::uint32_t value)
            {
                Words buffer = runtime.buffer<std::uint32_t>(size).value();
                std::fill(buffer.data(), buffer.data() + size, value);
                return buffer;
            }

            /** @brief Copies values into buffer from element offset on. */
            void load(const Words& buffer, const std::vector<std::uint32_t>& values,
                      std::size_t offset = 0)
            {
                std::copy(values.begin(), values.end(), buffer.data() + offset);
            }

            /** @brief Returns elements [offset, offset + count) of buffer. */
            std::vector<std::uint32_t> words_of(const Words& buffer, std::size_t offset,
                                                std::size_t count)
            {
                return std::vector<std::uint32_t>(buffer.data() + offset,
                                                  buffer.data() + offset + count);
            }

            /** @brief Holds the five buffers of one counting_sort() call. */
            struct SortOperands
            {
                Words keys;    /**< Keys, zero until a test loads its own. */
                Words indices; /**< Sorted order, K_CANARY until the sort writes it. */
                Words offsets; /**< Bucket starts, K_CANARY until the sort writes them. */
                Words total;   /**< Element count, K_CANARY until the sort writes it. */
                Words counts;  /**< Bucket sizes, K_CANARY until the sort writes them. */
            };

            /**
             * @brief Returns operands placed at window for count keys over key_range buckets.
             * @param tail Spare elements past each operand's window, left at their first value.
             */
            SortOperands operands_for(Runtime& runtime, std::size_t count, std::size_t key_range,
                                      std::size_t tail, const Window& window = Window{})
            {
                return SortOperands{
                    filled(runtime, std::max<std::size_t>(window.keys_offset + count + tail, 1),
                           0u),
                    filled(runtime, window.indices_offset + count + tail, K_CANARY),
                    filled(runtime, window.offsets_offset + key_range + 1 + tail, K_CANARY),
                    filled(runtime, window.total_offset + 1 + tail, K_CANARY),
                    filled(runtime, window.counts_offset + key_range + tail, K_CANARY)};
            }

            /// @brief Names the operand type of a host recorder.
            using HostWords = GraphPrimitives::HostBuffer<std::uint32_t>;

            /** @brief Returns a host buffer of size elements, each holding value. */
            HostWords host_filled(std::size_t size, std::uint32_t value)
            {
                HostWords buffer =
                    GraphPrimitives::HostPrimitiveRecorder::buffer<std::uint32_t>(size);
                std::fill(buffer.data(), buffer.data() + size, value);
                return buffer;
            }

            /** @brief Holds the five host buffers of one counting_sort() call on a host recorder. */
            struct HostSortOperands
            {
                HostWords keys;    /**< Keys. */
                HostWords indices; /**< Sorted order. */
                HostWords offsets; /**< Bucket starts. */
                HostWords total;   /**< Element count. */
                HostWords counts;  /**< Bucket sizes. */
            };

            /** @brief Returns host operands placed at window, sized as operands_for() sizes them. */
            HostSortOperands host_operands_for(std::size_t count, std::size_t key_range,
                                               std::size_t tail, const Window& window = Window{})
            {
                return HostSortOperands{
                    host_filled(std::max<std::size_t>(window.keys_offset + count + tail, 1), 0u),
                    host_filled(window.indices_offset + count + tail, K_CANARY),
                    host_filled(window.offsets_offset + key_range + 1 + tail, K_CANARY),
                    host_filled(window.total_offset + 1 + tail, K_CANARY),
                    host_filled(window.counts_offset + key_range + tail, K_CANARY)};
            }

            /** @brief Records counting_sort() over operands; the call must be accepted. */
            void record_sort(SushiRuntime::Recorder& recorder, const SortOperands& operands,
                             const LateCount& count, const LateCount& key_range,
                             const Words* counts = nullptr, const Window& window = Window{},
                             const std::optional<GraphPrimitives::PrimitiveOptions>& options =
                                 std::nullopt)
            {
                counting_sort(recorder, operands.keys, operands.indices, operands.offsets,
                              operands.total, count, key_range, counts, window, options)
                    .value();
            }

            /** @brief Records counting_sort() over operands on a host recorder; it must be accepted. */
            void record_sort(GraphPrimitives::HostPrimitiveRecorder& recorder,
                             const HostSortOperands& operands, const LateCount& count,
                             const LateCount& key_range, const HostWords* counts = nullptr,
                             const Window& window = Window{},
                             const std::optional<GraphPrimitives::PrimitiveOptions>& options =
                                 std::nullopt)
            {
                GraphPrimitives::counting_sort(recorder, operands.keys, operands.indices,
                                               operands.offsets, operands.total, count, key_range,
                                               counts, window, options)
                    .value();
            }

            /** @brief Holds what one sort wrote into the windows of its outputs. */
            struct SortOutcome
            {
                std::vector<std::uint32_t> indices;  /**< The count sorted indices. */
                std::vector<std::uint32_t> offsets;  /**< The key_range + 1 bucket starts. */
                std::vector<std::uint32_t> counts;   /**< The key_range bucket sizes. */
                std::uint32_t total = 0;             /**< The element count written. */
                std::vector<std::uint32_t> past_end; /**< First word past each output. */
            };

            /** @brief Returns the output windows of operands after a sort of count keys. */
            SortOutcome outcome_of(const SortOperands& operands, std::size_t count,
                                   std::size_t key_range, const Window& window = Window{})
            {
                return SortOutcome{words_of(operands.indices, window.indices_offset, count),
                                   words_of(operands.offsets, window.offsets_offset, key_range + 1),
                                   words_of(operands.counts, window.counts_offset, key_range),
                                   operands.total[window.total_offset],
                                   {}};
            }

            /**
             * @brief Returns the first word past the indices, offsets, counts and total windows.
             * @pre Each operand holds at least one element past its window.
             */
            std::vector<std::uint32_t> words_past(const SortOperands& operands, std::size_t count,
                                                  std::size_t key_range, const Window& window)
            {
                return {operands.indices[window.indices_offset + count],
                        operands.offsets[window.offsets_offset + key_range + 1],
                        operands.counts[window.counts_offset + key_range],
                        operands.total[window.total_offset + 1]};
            }

            /**
             * @brief Returns the word before each output window: indices, offsets, counts, total.
             * @pre Every output offset of window is at least one.
             */
            std::vector<std::uint32_t> words_before(const SortOperands& operands,
                                                    const Window& window)
            {
                return {operands.indices[window.indices_offset - 1],
                        operands.offsets[window.offsets_offset - 1],
                        operands.counts[window.counts_offset - 1],
                        operands.total[window.total_offset - 1]};
            }

            /**
             * @brief Sorts keys over key_range once, counts included, and returns what it wrote.
             *
             * Every output has one K_CANARY element past its end, which past_end returns.
             */
            SortOutcome sort_once(Runtime& runtime, const std::vector<std::uint32_t>& keys,
                                  std::size_t key_range)
            {
                const SortOperands operands = operands_for(runtime, keys.size(), key_range, 1);
                load(operands.keys, keys);

                auto graph = runtime.graph();
                record_sort(graph.recorder(), operands, LateCount::fixed(keys.size()),
                            LateCount::fixed(key_range), &operands.counts);
                graph.run().value();

                SortOutcome outcome = outcome_of(operands, keys.size(), key_range);
                outcome.past_end = words_past(operands, keys.size(), key_range, Window{});
                return outcome;
            }

            /** @brief Holds what each run of a series wrote beside what the host expects of it. */
            struct RunSeries
            {
                std::vector<std::vector<std::uint32_t>> indices;          /**< Indices per run. */
                std::vector<std::vector<std::uint32_t>> expected_indices; /**< Host indices. */
                std::vector<std::vector<std::uint32_t>> offsets;          /**< Offsets per run. */
                std::vector<std::vector<std::uint32_t>> expected_offsets; /**< Host offsets. */
                std::vector<std::uint32_t> totals;                        /**< Total per run. */
                std::vector<std::uint32_t> expected_totals;               /**< Key count per run. */
            };

            /** @brief Appends what operands hold after a sort of host_keys over key_range. */
            void append_run(RunSeries& series, const SortOperands& operands,
                            const std::vector<std::uint32_t>& host_keys, std::size_t key_range)
            {
                const HostReference reference = host_counting_sort(host_keys, key_range);
                series.indices.push_back(words_of(operands.indices, 0, host_keys.size()));
                series.expected_indices.push_back(reference.indices);
                series.offsets.push_back(words_of(operands.offsets, 0, key_range + 1));
                series.expected_offsets.push_back(reference.offsets);
                series.totals.push_back(operands.total[0]);
                series.expected_totals.push_back(static_cast<std::uint32_t>(host_keys.size()));
            }

            /** @brief Holds the live element count and key range of one run. */
            struct LiveSize
            {
                std::size_t count;     /**< Live element count. */
                std::size_t key_range; /**< Live key range. */
            };

            /** @brief Returns the seed of the keys a run of size sorts. */
            unsigned seed_of(const LiveSize& size)
            {
                return static_cast<unsigned>(size.count * 31 + size.key_range);
            }

            /** @brief Holds one sort recorded once with late-bound sizes and run at several. */
            class LateBoundSort final
            {
                public:
                    /** @brief Records one sort whose sizes read what run_each() sets. */
                    LateBoundSort(Runtime& runtime, std::size_t count_capacity,
                                  std::size_t key_capacity)
                        : live_(std::make_shared<LiveSize>(LiveSize{0, 0})),
                          operands_(operands_for(runtime, count_capacity, key_capacity, 0)),
                          graph_(runtime.graph())
                    {
                        const std::shared_ptr<LiveSize> live = live_;
                        record_sort(graph_.recorder(), operands_,
                                    LateCount::from_host(count_capacity,
                                                         [live]() { return live->count; }),
                                    LateCount::from_host(key_capacity,
                                                         [live]() { return live->key_range; }));
                    }

                    /** @brief Runs the sort once per size, each on seeded keys of that size. */
                    RunSeries run_each(const std::vector<LiveSize>& sizes)
                    {
                        RunSeries series;
                        for (const LiveSize& size : sizes)
                        {
                            *live_ = size;
                            const std::vector<std::uint32_t> host_keys =
                                seeded_keys(size.count, size.key_range, seed_of(size));
                            load(operands_.keys, host_keys);
                            graph_.run().value();
                            append_run(series, operands_, host_keys, size.key_range);
                        }
                        return series;
                    }

                    /** @brief Returns how many times the recorded graph compiled. */
                    [[nodiscard]] std::size_t compile_count() const
                    {
                        return graph_.compile_count();
                    }

                private:
                    std::shared_ptr<LiveSize> live_; /**< Size the providers read. */
                    SortOperands operands_;          /**< Buffers of the sort. */
                    SushiRuntime::Graph graph_;      /**< Graph holding the sort. */
            };

            /** @brief Returns the sizes the late-bound replay runs at: small, large, small. */
            std::vector<LiveSize> changing_sizes()
            {
                return {LiveSize{10, 4}, LiveSize{1500, 60000}, LiveSize{3, 2}};
            }

            /** @brief Holds the sort and the scan each run of a two-region DynamicGraph left. */
            struct RegionSeries
            {
                RunSeries sort;                                       /**< Sort per run. */
                std::vector<std::vector<std::uint32_t>> scan_offsets; /**< Scan per run. */
                std::vector<std::vector<std::uint32_t>> expected_scan_offsets; /**< Host scan. */
                std::vector<std::uint32_t> scan_totals;               /**< Scan total per run. */
                std::vector<std::uint32_t> expected_scan_totals;      /**< Host total per run. */
            };

            /** @brief Records a sort and a scan of its keys in two regions and runs two sizes. */
            RegionSeries replay_regions(Runtime& runtime)
            {
                constexpr std::size_t K_COUNT_CAPACITY = 700;
                constexpr std::size_t K_KEY_CAPACITY = 1000;
                const auto live = std::make_shared<LiveSize>(LiveSize{0, 0});
                const SortOperands operands =
                    operands_for(runtime, K_COUNT_CAPACITY, K_KEY_CAPACITY, 0);
                const Words scan_offsets = filled(runtime, K_COUNT_CAPACITY + 1, K_CANARY);
                const Words scan_total = filled(runtime, 1, K_CANARY);
                const LateCount count =
                    LateCount::from_host(K_COUNT_CAPACITY, [live]() { return live->count; });
                const LateCount key_range =
                    LateCount::from_host(K_KEY_CAPACITY, [live]() { return live->key_range; });

                auto world = runtime.dynamic_graph();
                record_sort(world.region(1).recorder(), operands, count, key_range);
                prefix_scan(world.region(2).recorder(), operands.keys, scan_offsets, scan_total,
                            count).value();

                RegionSeries series;
                for (const LiveSize& size : {LiveSize{300, 200}, LiveSize{700, 1000}})
                {
                    *live = size;
                    const std::vector<std::uint32_t> host_keys =
                        seeded_keys(size.count, size.key_range, seed_of(size));
                    load(operands.keys, host_keys);
                    world.run().value();

                    append_run(series.sort, operands, host_keys, size.key_range);
                    const std::vector<std::uint32_t> sums = exclusive_sums(host_keys);
                    series.scan_offsets.push_back(words_of(scan_offsets, 0, size.count + 1));
                    series.expected_scan_offsets.push_back(sums);
                    series.scan_totals.push_back(scan_total[0]);
                    series.expected_scan_totals.push_back(sums.back());
                }
                return series;
            }

            /** @brief Runs one fixed sort of 500 keys four times, on new keys each run. */
            RunSeries sort_each_step(Runtime& runtime)
            {
                constexpr std::uint32_t K_STEPS = 4;
                const SortOperands operands =
                    operands_for(runtime, K_STEP_COUNT, K_STEP_KEY_RANGE, 0);

                auto graph = runtime.graph();
                record_sort(graph.recorder(), operands, LateCount::fixed(K_STEP_COUNT),
                            LateCount::fixed(K_STEP_KEY_RANGE));

                RunSeries series;
                for (std::uint32_t step = 0; step < K_STEPS; ++step)
                {
                    const std::vector<std::uint32_t> host_keys = step_keys(step);
                    load(operands.keys, host_keys);
                    graph.run().value();
                    append_run(series, operands, host_keys, K_STEP_KEY_RANGE);
                }
                return series;
            }

            /** @brief Holds the dispatches of a captured sort and the rules it recorded. */
            struct CapturedSort
            {
                std::vector<NodeEventRecord> events; /**< One record per dispatch, in order. */
                std::size_t rules = 0;               /**< Rules the call recorded. */
            };

            /** @brief Runs a sort of 300 keys over 50 buckets, counts included, for steps steps. */
            CapturedSort capture_sort(Runtime& runtime, std::size_t steps)
            {
                const SortOperands operands =
                    operands_for(runtime, K_CAPTURED_COUNT, K_CAPTURED_KEY_RANGE, 0);
                load(operands.keys, cycled_keys(K_CAPTURED_COUNT, K_CAPTURED_KEY_RANGE));

                auto graph = runtime.graph();
                const RunObserverCapture capture(runtime);
                record_sort(graph.recorder(), operands, LateCount::fixed(K_CAPTURED_COUNT),
                            LateCount::fixed(K_CAPTURED_KEY_RANGE), &operands.counts);
                graph.run(steps).value();
                return CapturedSort{capture.records(), graph.recorder().rule_count()};
            }

            /** @brief Returns how many distinct node indices events carry. */
            std::size_t distinct_node_indices(const std::vector<NodeEventRecord>& events)
            {
                std::set<std::uint32_t> indices;
                for (const NodeEventRecord& record : events)
                    indices.insert(record.node_index);
                return indices.size();
            }

            /** @brief Returns the dispatches of each invocation, up to the highest seen. */
            std::vector<std::size_t>
            dispatches_per_invocation(const std::vector<NodeEventRecord>& events)
            {
                std::vector<std::size_t> per_invocation;
                for (const NodeEventRecord& record : events)
                {
                    if (record.invocation >= per_invocation.size())
                        per_invocation.resize(record.invocation + std::size_t{1}, 0);
                    ++per_invocation[record.invocation];
                }
                return per_invocation;
            }

            /** @brief Holds what a sort over windows and one over compact buffers wrote. */
            struct WindowRun
            {
                SortOutcome windowed;             /**< Output windows of the windowed sort. */
                SortOutcome compact;              /**< Outputs of the compact sort. */
                std::vector<std::uint32_t> around; /**< Words before, then past, each window. */
            };

            /** @brief Sorts 3000 seeded keys at K_WINDOW and on compact buffers. */
            WindowRun sort_windowed_and_compact(Runtime& runtime)
            {
                constexpr std::size_t K_COUNT = 3000;
                constexpr std::size_t K_KEY_RANGE = 1000;
                const std::vector<std::uint32_t> host = seeded_keys(K_COUNT, K_KEY_RANGE, 18u);
                const SortOperands wide =
                    operands_for(runtime, K_COUNT, K_KEY_RANGE, K_WINDOW_TAIL, K_WINDOW);
                const SortOperands compact = operands_for(runtime, K_COUNT, K_KEY_RANGE, 0);
                load(wide.keys, host, K_WINDOW.keys_offset);
                load(compact.keys, host);

                auto graph = runtime.graph();
                record_sort(graph.recorder(), wide, LateCount::fixed(K_COUNT),
                            LateCount::fixed(K_KEY_RANGE), &wide.counts, K_WINDOW);
                record_sort(graph.recorder(), compact, LateCount::fixed(K_COUNT),
                            LateCount::fixed(K_KEY_RANGE), &compact.counts);
                graph.run().value();

                std::vector<std::uint32_t> around = words_before(wide, K_WINDOW);
                const std::vector<std::uint32_t> past =
                    words_past(wide, K_COUNT, K_KEY_RANGE, K_WINDOW);
                around.insert(around.end(), past.begin(), past.end());
                return WindowRun{outcome_of(wide, K_COUNT, K_KEY_RANGE, K_WINDOW),
                                 outcome_of(compact, K_COUNT, K_KEY_RANGE), around};
            }

            /** @brief Returns the stage names a call over key_range records, in order. */
            std::vector<std::string> stages_for(std::size_t key_range, bool with_counts)
            {
                std::vector<std::string> stages = {"resolve"};
                const std::size_t digits = GraphPrimitives::Detail::radix_digit_count(key_range);
                for (std::size_t d = 0; d < digits; ++d)
                    stages.insert(stages.end(),
                                  {"histogram", "column_scan", "bin_base", "placement"});
                stages.push_back("offsets");
                if (with_counts)
                    stages.push_back("counts");
                return stages;
            }

            /** @brief Returns a host recorder holding one sort of 8 keys over 4 buckets with options. */
            GraphPrimitives::HostPrimitiveRecorder labelled_sort(
                const HostSortOperands& operands, bool with_counts,
                const std::optional<GraphPrimitives::PrimitiveOptions>& options)
            {
                GraphPrimitives::HostPrimitiveRecorder recorder;
                record_sort(recorder, operands, LateCount::fixed(8),
                            LateCount::fixed(K_LABEL_KEY_RANGE),
                            with_counts ? &operands.counts : nullptr, Window{}, options);
                return recorder;
            }

            /** @brief Adds a node that packs the outputs of the device-sized sort. */
            void add_packer(SushiRuntime::Graph& graph, const Words& indices, const Words& offsets,
                            const Words& total, const Words& packed)
            {
                const std::uint32_t* ip = indices.data();
                const std::uint32_t* op = offsets.data();
                const std::uint32_t* tp = total.data();
                std::uint32_t* pp = packed.data();
                graph.add(K_DEVICE_PACKED, SushiRuntime::Reads(indices, offsets, total),
                          SushiRuntime::Writes(packed),
                          [ip, op, tp, pp](std::size_t i)
                          {
                              if (i < K_DEVICE_COUNT_CAPACITY)
                                  pp[i] = ip[i];
                              else if (i < K_DEVICE_COUNT_CAPACITY + K_DEVICE_BUCKETS + 1)
                                  pp[i] = op[i - K_DEVICE_COUNT_CAPACITY];
                              else
                                  pp[i] = tp[0];
                          });
            }

            /// @brief Names a key range capacity and the nodes a call over it records.
            struct NodeCountRow
            {
                /// @brief Holds the key range capacity.
                std::size_t key_capacity;
                /// @brief Holds the nodes recorded: the resolver, four per digit and the offsets.
                std::size_t nodes;
            };

            /// @brief Names one live key range below a capacity of 2^17.
            struct LiveRangeRow
            {
                /// @brief Holds the live key range.
                std::size_t key_range;
            };

            /// @brief Names one pass of a captured sort and the width it launches at.
            struct WidthRow
            {
                /// @brief Holds the pass, in dispatch order.
                std::size_t pass;
                /// @brief Holds the width the pass launches and works over.
                std::uint32_t width;
            };

            /// @brief Names a region of one operand and whether a writer of it waits.
            struct ProbeRow
            {
                /// @brief Holds the operand the region lies in.
                HostWords HostSortOperands::*operand;
                /// @brief Holds the first element of the region.
                std::size_t offset;
                /// @brief Holds the elements of the region.
                std::size_t count;
                /// @brief Holds whether a writer of the region waits for the sort.
                bool ordered;
            };

        } // namespace

        /** @brief Tests that duplicate keys come out in the order std::stable_sort gives. */
        TEST(Medium_GraphPrimitivesCountingSort, StableAgainstHostReferenceOnDuplicateKeys)
        {
            const std::vector<std::uint32_t> keys = duplicate_keys();
            const HostReference reference = host_counting_sort(keys, K_DUPLICATE_KEY_RANGE);
            ASSERT_NE(unstable_order(keys), reference.indices)
                << "the input must exercise stability, or this test proves nothing";

            auto runtime = Runtime::create().value();
            EXPECT_EQ(sort_once(runtime, keys, K_DUPLICATE_KEY_RANGE).indices, reference.indices);
        }

        /** @brief Tests that the bucket offsets of duplicate keys equal the host's. */
        TEST(Medium_GraphPrimitivesCountingSort, DuplicateKeysGiveTheHostBucketOffsets)
        {
            const std::vector<std::uint32_t> keys = duplicate_keys();

            auto runtime = Runtime::create().value();
            EXPECT_EQ(sort_once(runtime, keys, K_DUPLICATE_KEY_RANGE).offsets,
                      host_counting_sort(keys, K_DUPLICATE_KEY_RANGE).offsets);
        }

        /** @brief Tests that the bucket counts of duplicate keys equal the host's. */
        TEST(Medium_GraphPrimitivesCountingSort, DuplicateKeysGiveTheHostBucketCounts)
        {
            const std::vector<std::uint32_t> keys = duplicate_keys();

            auto runtime = Runtime::create().value();
            EXPECT_EQ(sort_once(runtime, keys, K_DUPLICATE_KEY_RANGE).counts,
                      host_counting_sort(keys, K_DUPLICATE_KEY_RANGE).counts);
        }

        /** @brief Tests that the total holds the element count. */
        TEST(Medium_GraphPrimitivesCountingSort, TotalHoldsTheElementCount)
        {
            auto runtime = Runtime::create().value();
            EXPECT_EQ(sort_once(runtime, duplicate_keys(), K_DUPLICATE_KEY_RANGE).total,
                      K_DUPLICATE_COUNT);
        }

        /** @brief Tests that the sort writes nothing past indices, offsets, counts or total. */
        TEST(Medium_GraphPrimitivesCountingSort, NothingIsWrittenPastTheOutputs)
        {
            auto runtime = Runtime::create().value();
            EXPECT_EQ(sort_once(runtime, duplicate_keys(), K_DUPLICATE_KEY_RANGE).past_end,
                      std::vector<std::uint32_t>(4, K_CANARY));
        }

        /** @brief Tests stability across digits when every key shares the same low byte. */
        TEST(Medium_GraphPrimitivesCountingSort, StableAcrossDigitsWhenTheLowDigitTies)
        {
            constexpr std::size_t K_KEY_CAPACITY = std::size_t{1} << 16;
            const std::vector<std::uint32_t> keys = low_digit_tied_keys();
            const HostReference reference = host_counting_sort(keys, K_KEY_CAPACITY);
            ASSERT_NE(unstable_order(keys), reference.indices)
                << "the input must exercise stability, or this test proves nothing";

            auto runtime = Runtime::create().value();
            const SortOutcome sorted = sort_once(runtime, keys, K_KEY_CAPACITY);

            EXPECT_EQ(sorted.indices, reference.indices);
            EXPECT_EQ(sorted.offsets, reference.offsets);
            EXPECT_EQ(sorted.total, keys.size());
        }

        /** @brief Tests 2^17 seeded elements over a 2^20 key range against the host reference. */
        TEST(Medium_GraphPrimitivesCountingSort, LargeRandomCaseAgainstHostReference)
        {
            // 2^17 keys fill 512 tiles, more than the 256 bins of a digit, over three digits.
            constexpr std::size_t K_COUNT = std::size_t{1} << 17;
            constexpr std::size_t K_KEY_RANGE = std::size_t{1} << 20;
            const std::vector<std::uint32_t> keys = seeded_keys(K_COUNT, K_KEY_RANGE, 20260918);
            const HostReference reference = host_counting_sort(keys, K_KEY_RANGE);

            auto runtime = Runtime::create().value();
            const SortOutcome sorted = sort_once(runtime, keys, K_KEY_RANGE);

            EXPECT_EQ(sorted.indices, reference.indices);
            EXPECT_EQ(sorted.offsets, reference.offsets);
            EXPECT_EQ(sorted.total, K_COUNT);
        }

        /** @brief Tests that equal keys keep the input order, since every element ties. */
        TEST(Medium_GraphPrimitivesCountingSort, AllKeysEqualKeepsInputOrder)
        {
            constexpr std::size_t K_COUNT = 777;

            auto runtime = Runtime::create().value();
            const SortOutcome sorted =
                sort_once(runtime, std::vector<std::uint32_t>(K_COUNT, 2u), 4);

            EXPECT_EQ(sorted.indices, ascending_words(K_COUNT));
            EXPECT_EQ(sorted.offsets, (std::vector<std::uint32_t>{0, 0, 0, 777, 777}));
            EXPECT_EQ(sorted.total, K_COUNT);
        }

        /** @brief Tests one element, the smallest input that runs every pass. */
        TEST(Medium_GraphPrimitivesCountingSort, OneElement)
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
        TEST(Medium_GraphPrimitivesCountingSort, ZeroElementsWritesZeroEverywhere)
        {
            constexpr std::size_t K_KEY_RANGE = 16;

            auto runtime = Runtime::create().value();
            const SortOutcome sorted = sort_once(runtime, {}, K_KEY_RANGE);

            EXPECT_EQ(sorted.total, 0u);
            EXPECT_EQ(sorted.offsets, std::vector<std::uint32_t>(K_KEY_RANGE + 1, 0u));
            EXPECT_EQ(sorted.counts, std::vector<std::uint32_t>(K_KEY_RANGE, 0u));
            EXPECT_EQ(sorted.past_end, std::vector<std::uint32_t>(4, K_CANARY));
        }

        /** @brief Tests the largest key of a 2^20 range, which sets bit 19. */
        TEST(Medium_GraphPrimitivesCountingSort, MaximumKeyOfA20BitRange)
        {
            constexpr std::size_t K_KEY_RANGE = std::size_t{1} << 20;
            constexpr auto K_LARGEST = static_cast<std::uint32_t>(K_KEY_RANGE - 1);

            auto runtime = Runtime::create().value();
            const SortOutcome sorted =
                sort_once(runtime, {K_LARGEST, 0, K_LARGEST, 1, 0}, K_KEY_RANGE);

            EXPECT_EQ(sorted.indices, (std::vector<std::uint32_t>{1, 4, 3, 0, 2}));
            EXPECT_EQ(sorted.offsets[K_KEY_RANGE - 1], 3u);
            EXPECT_EQ(sorted.offsets[K_KEY_RANGE], 5u);
        }

        /// @brief Records a sort over one key range capacity.
        class Small_GraphPrimitivesCountingSortCapacity
            : public ::testing::TestWithParam<NodeCountRow>
        {
        };

        /** @brief Tests that each extra digit the key capacity needs records four more nodes. */
        TEST_P(Small_GraphPrimitivesCountingSortCapacity, DigitCountFollowsKeyRangeCapacity)
        {
            constexpr std::size_t K_COUNT = 10;
            const NodeCountRow& row = GetParam();

            const HostSortOperands operands =
                host_operands_for(K_COUNT, row.key_capacity, 0);
            GraphPrimitives::HostPrimitiveRecorder recorder;
            record_sort(recorder, operands, LateCount::fixed(K_COUNT),
                        LateCount::fixed(row.key_capacity));

            EXPECT_EQ(recorder.nodes().size(), row.nodes);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_GraphPrimitivesCountingSortCapacity,
            ::testing::Values(NodeCountRow{std::size_t{1} << 8, 1 * K_NODES_PER_DIGIT + 2},
                              NodeCountRow{std::size_t{1} << 16, 2 * K_NODES_PER_DIGIT + 2},
                              NodeCountRow{std::size_t{1} << 20, 3 * K_NODES_PER_DIGIT + 2}));

        /** @brief Tests that a run dispatches each recorded node of the sort once. */
        TEST(Medium_GraphPrimitivesCountingSort, EachRecordedNodeDispatchesOncePerRun)
        {
            constexpr std::size_t K_COUNT = 10;
            constexpr std::size_t K_KEY_CAPACITY = std::size_t{1} << 20;

            auto runtime = Runtime::create().value();
            const SortOperands operands = operands_for(runtime, K_COUNT, K_KEY_CAPACITY, 0);
            auto graph = runtime.graph();
            const RunObserverCapture capture(runtime);
            record_sort(graph.recorder(), operands, LateCount::fixed(K_COUNT),
                        LateCount::fixed(K_KEY_CAPACITY));
            graph.run().value();

            EXPECT_EQ(capture.records().size(), graph.recorder().node_count());
        }

        /** @brief Tests that each run of a late-bound sort sorts at the sizes it reads. */
        TEST(Medium_GraphPrimitivesCountingSort, LateBoundChangingCountAndKeyRangeSortsEveryRun)
        {
            auto runtime = Runtime::create().value();
            LateBoundSort sort(runtime, 2000, std::size_t{1} << 16);
            const RunSeries series = sort.run_each(changing_sizes());

            EXPECT_EQ(series.indices, series.expected_indices);
            EXPECT_EQ(series.offsets, series.expected_offsets);
            EXPECT_EQ(series.totals, series.expected_totals);
        }

        /** @brief Tests that a count and a key range that change between runs keep one compile. */
        TEST(Medium_GraphPrimitivesCountingSort, LateBoundChangingCountAndKeyRangeKeepsOneCompile)
        {
            auto runtime = Runtime::create().value();
            LateBoundSort sort(runtime, 2000, std::size_t{1} << 16);
            static_cast<void>(sort.run_each(changing_sizes()));

            EXPECT_EQ(sort.compile_count(), 1u)
                << "a live count or key range change between runs must not trigger a recompile";
        }

        /// @brief Sorts at one live key range below a capacity of 2^17.
        class Medium_GraphPrimitivesCountingSortLiveKeyRange
            : public ::testing::TestWithParam<LiveRangeRow>
        {
        };

        /** @brief Tests that a live key range of one, two or three digits ends in indices. */
        TEST_P(Medium_GraphPrimitivesCountingSortLiveKeyRange,
               LiveKeyRangeBelowCapacityLandsInIndices)
        {
            // 2^17 needs three digits; the rows below it need three, two, one and one.
            constexpr std::size_t K_COUNT = 3000;
            constexpr std::size_t K_KEY_CAPACITY = std::size_t{1} << 17;

            auto runtime = Runtime::create().value();
            LateBoundSort sort(runtime, K_COUNT, K_KEY_CAPACITY);
            const RunSeries series = sort.run_each({LiveSize{K_COUNT, GetParam().key_range}});

            EXPECT_EQ(series.indices, series.expected_indices);
            EXPECT_EQ(series.offsets, series.expected_offsets);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_GraphPrimitivesCountingSortLiveKeyRange,
                                 ::testing::Values(LiveRangeRow{100000}, LiveRangeRow{40000},
                                                   LiveRangeRow{200}, LiveRangeRow{3}));

        /** @brief Tests that the digits above the live key range do not run. */
        TEST(Medium_GraphPrimitivesCountingSort, DigitsAboveTheLiveKeyRangeDoNotRun)
        {
            constexpr std::size_t K_COUNT = 4;
            constexpr std::size_t K_KEY_CAPACITY = std::size_t{1} << 16;

            auto runtime = Runtime::create().value();
            const SortOperands operands = operands_for(runtime, K_COUNT, K_KEY_CAPACITY, 0);
            // One digit puts 0x0101 between 0 and 2; a second digit would put it last.
            load(operands.keys, {0x0101, 2, 0, 3});

            auto graph = runtime.graph();
            record_sort(graph.recorder(), operands, LateCount::fixed(K_COUNT),
                        LateCount::from_host(K_KEY_CAPACITY, []() { return std::size_t{4}; }));
            graph.run().value();

            EXPECT_EQ(words_of(operands.indices, 0, K_COUNT),
                      (std::vector<std::uint32_t>{2, 0, 1, 3}));
        }

        /** @brief Tests that a DynamicGraph region records the sort as a Graph does. */
        TEST(Medium_GraphPrimitivesCountingSort, RegionsRecordCountingSort)
        {
            auto runtime = Runtime::create().value();
            const RegionSeries series = replay_regions(runtime);

            EXPECT_EQ(series.sort.indices, series.sort.expected_indices);
            EXPECT_EQ(series.sort.offsets, series.sort.expected_offsets);
        }

        /** @brief Tests that a DynamicGraph region records the scan beside the sort. */
        TEST(Medium_GraphPrimitivesCountingSort, RegionsRecordPrefixScan)
        {
            auto runtime = Runtime::create().value();
            const RegionSeries series = replay_regions(runtime);

            EXPECT_EQ(series.scan_offsets, series.expected_scan_offsets);
            EXPECT_EQ(series.scan_totals, series.expected_scan_totals);
        }

        /** @brief Tests that each run sorts the keys as they stand when it runs. */
        TEST(Medium_GraphPrimitivesCountingSort, EachRunSortsItsOwnKeys)
        {
            auto runtime = Runtime::create().value();
            const RunSeries series = sort_each_step(runtime);

            EXPECT_EQ(series.indices, series.expected_indices);
            EXPECT_EQ(series.offsets, series.expected_offsets);
        }

        /** @brief Tests that every node of one call shares one node_index on every step. */
        TEST(Medium_GraphPrimitivesCountingSort, SharesOneNodeIndexAcrossStepsAndPasses)
        {
            auto runtime = Runtime::create().value();
            const CapturedSort captured = capture_sort(runtime, 4);

            EXPECT_EQ(captured.rules, 1u);
            EXPECT_EQ(distinct_node_indices(captured.events), 1u)
                << "every pass of one counting_sort() call must share one node_index";
        }

        /** @brief Tests that every step of a multi-step run dispatches every pass once. */
        TEST(Medium_GraphPrimitivesCountingSort, EveryStepRunsEveryPass)
        {
            // One digit records the resolver, four digit passes, the offsets and the counts.
            constexpr std::size_t K_EXPECTED_PASSES = K_NODES_PER_DIGIT + 3;
            constexpr std::size_t K_STEPS = 4;

            auto runtime = Runtime::create().value();
            const CapturedSort captured = capture_sort(runtime, K_STEPS);

            EXPECT_EQ(dispatches_per_invocation(captured.events),
                      std::vector<std::size_t>(K_STEPS, K_EXPECTED_PASSES));
        }

        /// @brief Reads the width one pass of a recorded sort launches at.
        class Small_GraphPrimitivesCountingSortNodeWidth
            : public ::testing::TestWithParam<WidthRow>
        {
        };

        /** @brief Tests that each node launches at the width its pass has in the plan. */
        TEST_P(Small_GraphPrimitivesCountingSortNodeWidth, NodeWidthsMatchThePlan)
        {
            const WidthRow& row = GetParam();

            const HostSortOperands operands =
                host_operands_for(K_CAPTURED_COUNT, K_CAPTURED_KEY_RANGE, 0);
            GraphPrimitives::HostPrimitiveRecorder recorder;
            record_sort(recorder, operands, LateCount::fixed(K_CAPTURED_COUNT),
                        LateCount::fixed(K_CAPTURED_KEY_RANGE), &operands.counts);

            ASSERT_EQ(recorder.nodes().size(), 7u);
            EXPECT_EQ(recorder.nodes()[row.pass].width, std::size_t{row.width});
        }

        // Resolver, histogram, column scan, bin base, placement, offsets, counts.
        INSTANTIATE_TEST_SUITE_P(
            , Small_GraphPrimitivesCountingSortNodeWidth,
            ::testing::Values(
                WidthRow{0, 1u}, WidthRow{1, 2u},
                WidthRow{2,
                         static_cast<std::uint32_t>(GraphPrimitives::Detail::K_RADIX_DIGIT_BINS)},
                WidthRow{3, 1u}, WidthRow{4, 2u}, WidthRow{5, 51u}, WidthRow{6, 50u}));

        /** @brief Tests that a live element count above its capacity fails the run. */
        TEST(Medium_GraphPrimitivesCountingSort, LateBoundCountAboveCapacityFailsTheRun)
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
        TEST(Medium_GraphPrimitivesCountingSort, LateBoundKeyRangeAboveCapacityFailsTheRun)
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

        /** @brief Tests counting_sort() with a device count and a host key range on both layers. */
        TEST(Medium_GraphPrimitivesDeviceSized, CountingSortChainMatchesScheduler)
        {
            using namespace DeviceSized;

            expect_chain_matches_scheduler([](Runtime& runtime)
            {
                Harness harness(runtime, K_DEVICE_COUNT_CAPACITY);
                const SortOperands operands =
                    operands_for(runtime, K_DEVICE_COUNT_CAPACITY, K_DEVICE_KEY_CAPACITY, 0);
                const Words packed = filled(runtime, K_DEVICE_PACKED, 0u);
                load(operands.keys, device_sized_keys());
                load(operands.indices, std::vector<std::uint32_t>(K_DEVICE_COUNT_CAPACITY, 0u));

                // The live key range changes every step, so one, two and three digits all run.
                const std::array<std::size_t, 4> ranges{K_DEVICE_BUCKETS, 300, 70000,
                                                        K_DEVICE_KEY_CAPACITY};
                auto tick = std::make_shared<std::size_t>(0);
                const LateCount key_range = LateCount::from_host(
                    K_DEVICE_KEY_CAPACITY,
                    [ranges, tick] { return ranges[(*tick)++ % ranges.size()]; });

                auto graph = runtime.graph();
                harness.add_producer(graph);
                record_sort(graph.recorder(), operands,
                            harness.device_count(K_DEVICE_COUNT_CAPACITY), key_range);
                add_packer(graph, operands.indices, operands.offsets, operands.total, packed);
                harness.add_digest(graph, packed, K_DEVICE_PACKED);
                return harness.record(graph);
            });
        }

        /** @brief Tests that a sort over windows writes what one over compact buffers writes. */
        TEST(Medium_GraphPrimitivesCountingSort, WindowMatchesACompactBufferExactly)
        {
            auto runtime = Runtime::create().value();
            const WindowRun run = sort_windowed_and_compact(runtime);

            EXPECT_EQ(run.windowed.indices, run.compact.indices);
            EXPECT_EQ(run.windowed.offsets, run.compact.offsets);
            EXPECT_EQ(run.windowed.counts, run.compact.counts);
            EXPECT_EQ(run.windowed.total, run.compact.total);
        }

        /** @brief Tests that a sort over windows writes nothing just before or past each window. */
        TEST(Medium_GraphPrimitivesCountingSort, WindowLeavesTheElementsAroundItUntouched)
        {
            auto runtime = Runtime::create().value();
            const WindowRun run = sort_windowed_and_compact(runtime);

            EXPECT_EQ(run.around, std::vector<std::uint32_t>(8, K_CANARY));
        }

        /// @brief Probes one region of one operand of a sort at K_WINDOW.
        class Small_GraphPrimitivesCountingSortWindowProbe
            : public ::testing::TestWithParam<ProbeRow>
        {
        };

        /** @brief Tests that a writer of an operand waits only when it hits the window. */
        TEST_P(Small_GraphPrimitivesCountingSortWindowProbe, WindowOrdersOnlyOverlappingWriters)
        {
            const ProbeRow& row = GetParam();

            const HostSortOperands operands =
                host_operands_for(K_PROBE_COUNT, K_PROBE_KEY_RANGE, K_WINDOW_TAIL, K_WINDOW);
            GraphPrimitives::HostPrimitiveRecorder recorder;
            record_sort(recorder, operands, LateCount::fixed(K_PROBE_COUNT),
                        LateCount::fixed(K_PROBE_KEY_RANGE), &operands.counts, K_WINDOW);
            const std::size_t waits = recorder.nodes_touching(
                WindowProbe::region_of(operands.*row.operand, row.offset, row.count));

            EXPECT_EQ(waits > 0, row.ordered) << "the writer waits for " << waits << " nodes";
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_GraphPrimitivesCountingSortWindowProbe,
            ::testing::Values(
                ProbeRow{&HostSortOperands::keys, 0, K_WINDOW.keys_offset, false},
                ProbeRow{&HostSortOperands::keys, K_WINDOW.keys_offset + K_PROBE_COUNT,
                         K_WINDOW_TAIL, false},
                ProbeRow{&HostSortOperands::keys, K_WINDOW.keys_offset, 1, true},
                ProbeRow{&HostSortOperands::indices, K_WINDOW.indices_offset + K_PROBE_COUNT,
                         K_WINDOW_TAIL, false},
                ProbeRow{&HostSortOperands::indices, K_WINDOW.indices_offset + K_PROBE_COUNT - 1, 1,
                         true},
                ProbeRow{&HostSortOperands::offsets, 0, K_WINDOW.offsets_offset, false},
                ProbeRow{&HostSortOperands::offsets, K_WINDOW.offsets_offset + K_PROBE_KEY_RANGE, 1,
                         true},
                ProbeRow{&HostSortOperands::total, K_WINDOW.total_offset + 1, 1, false},
                ProbeRow{&HostSortOperands::total, K_WINDOW.total_offset, 1, true},
                ProbeRow{&HostSortOperands::counts, K_WINDOW.counts_offset + K_PROBE_KEY_RANGE,
                         K_WINDOW_TAIL, false},
                ProbeRow{&HostSortOperands::counts, K_WINDOW.counts_offset, 1, true}));

        /// @brief Records a named sort with or without counts.
        class Small_GraphPrimitivesCountingSortNamedCall : public ::testing::TestWithParam<bool>
        {
        };

        /** @brief Tests that a named call names its nodes "sort.<stage>" in recording order. */
        TEST_P(Small_GraphPrimitivesCountingSortNamedCall, NamedCallNamesItsNodesByStage)
        {
            const bool with_counts = GetParam();
            const GraphPrimitives::PrimitiveOptions options{"sort", Determinism::Bitwise};

            const HostSortOperands operands =
                host_operands_for(8, K_LABEL_KEY_RANGE, 0);
            const GraphPrimitives::HostPrimitiveRecorder recorded =
                labelled_sort(operands, with_counts, options);

            EXPECT_EQ(HostNodes::names_of(recorded),
                      PlanLabels::staged("sort", stages_for(K_LABEL_KEY_RANGE, with_counts)));
        }

        /** @brief Tests that a named call gives every node the class its options state. */
        TEST_P(Small_GraphPrimitivesCountingSortNamedCall, NamedCallStatesTheClassOfEveryNode)
        {
            const bool with_counts = GetParam();
            const GraphPrimitives::PrimitiveOptions options{"sort", Determinism::Bitwise};

            const HostSortOperands operands =
                host_operands_for(8, K_LABEL_KEY_RANGE, 0);
            const GraphPrimitives::HostPrimitiveRecorder recorded =
                labelled_sort(operands, with_counts, options);

            EXPECT_EQ(HostNodes::determinisms_of(recorded),
                      std::vector<Determinism>(stages_for(K_LABEL_KEY_RANGE, with_counts).size(),
                                               Determinism::Bitwise));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesCountingSortNamedCall, ::testing::Bool());

        /** @brief Tests that a call with no options keeps its nodes unnamed. */
        TEST(Small_GraphPrimitivesCountingSort, UnnamedCallKeepsItsNodesUnnamed)
        {
            const HostSortOperands operands =
                host_operands_for(8, K_LABEL_KEY_RANGE, 0);
            const GraphPrimitives::HostPrimitiveRecorder recorded =
                labelled_sort(operands, false, std::nullopt);

            EXPECT_EQ(HostNodes::names_of(recorded),
                      std::vector<std::string>(stages_for(K_LABEL_KEY_RANGE, false).size(),
                                               std::string()));
        }

        /** @brief Tests that a call with no options leaves the class of every node Unstated. */
        TEST(Small_GraphPrimitivesCountingSort, UnnamedCallLeavesEveryClassUnstated)
        {
            const HostSortOperands operands =
                host_operands_for(8, K_LABEL_KEY_RANGE, 0);
            const GraphPrimitives::HostPrimitiveRecorder recorded =
                labelled_sort(operands, false, std::nullopt);

            EXPECT_EQ(HostNodes::determinisms_of(recorded),
                      std::vector<Determinism>(stages_for(K_LABEL_KEY_RANGE, false).size(),
                                               Determinism::Unstated));
        }

    } // namespace Test
} // namespace SushiBLAS
