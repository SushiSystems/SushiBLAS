/**************************************************************************/
/* test_device_sized_primitives.cpp                                       */
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
 * @file test_device_sized_primitives.cpp
 * @brief Tests that each primitive gives one result on the chain and on the scheduler.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/graph/graph.hpp>
#include <SushiRuntime/api/runtime/runtime.hpp>
#include <SushiRuntime/api/runtime/runtime_config.hpp>
#include <SushiRuntime/api/vocabulary/dependencies.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>
#include <SushiRuntime/core/error.hpp>
#include <SushiRuntime/report/run_report.hpp>

#include <SushiBLAS/graph_primitives/combiners.hpp>
#include <SushiBLAS/graph_primitives/counting_sort.hpp>
#include <SushiBLAS/graph_primitives/prefix_scan.hpp>
#include <SushiBLAS/graph_primitives/reduce.hpp>
#include <SushiBLAS/graph_primitives/segmented_reduce.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using GraphPrimitives::Sum;
            using SushiRuntime::Buffer;
            using SushiRuntime::ExecutionLayer;
            using SushiRuntime::Graph;
            using SushiRuntime::LateCount;
            using SushiRuntime::Reads;
            using SushiRuntime::Runtime;
            using SushiRuntime::Writes;

            /// @brief Holds the steps one recording runs.
            constexpr std::size_t K_STEPS = 48;

            /// @brief Holds the offset basis of the 32-bit FNV-1a digest.
            constexpr std::uint32_t K_DIGEST_BASIS = 2166136261u;

            /// @brief Holds the prime of the 32-bit FNV-1a digest.
            constexpr std::uint32_t K_DIGEST_PRIME = 16777619u;

            /** @brief Holds the per-step digests of one run and the layer it took. */
            struct Recording
            {
                std::vector<std::uint32_t> digests;                /**< One word per step. */
                ExecutionLayer layer = ExecutionLayer::Scheduler; /**< Layer the run took. */
            };

            /**
             * @brief Returns a live-count schedule over a capacity that hits 0 and the capacity.
             * @return Eleven counts, cycled by the producer.
             */
            std::vector<std::uint32_t> schedule_for(std::size_t capacity)
            {
                const auto cap = static_cast<std::uint32_t>(capacity);
                const auto clamp = [cap](std::uint32_t v) { return v > cap ? cap : v; };
                return {cap, 0u, clamp(1u), cap / 2u + (cap > 1u ? 1u : 0u), cap, 0u, cap / 3u,
                        cap > 0u ? cap - 1u : 0u, clamp(cap / 7u + 3u), cap, clamp(2u)};
            }

            /** @brief Holds the step counter, the live count and the digests of one graph. */
            struct Harness
            {
                Buffer<std::uint32_t> counter;  /**< Residency::Host live count per step. */
                Buffer<std::uint32_t> step;     /**< Steps the producer has run. */
                Buffer<std::uint32_t> schedule; /**< Live count per step, cycled. */
                Buffer<std::uint32_t> history;  /**< One digest per step. */

                /** @brief Allocates the buffers and writes the schedule of a capacity. */
                Harness(Runtime& runtime, std::size_t capacity)
                    : counter(runtime.buffer<std::uint32_t>(
                          1, SushiRuntime::DeviceIndex{0},
                          SushiRuntime::Residency::Host).value()),
                      step(runtime.buffer<std::uint32_t>(1).value()),
                      schedule(runtime.buffer<std::uint32_t>(schedule_for(capacity).size())
                                   .value()),
                      history(runtime.buffer<std::uint32_t>(K_STEPS).value())
                {
                    const std::vector<std::uint32_t> counts = schedule_for(capacity);
                    for (std::size_t i = 0; i < counts.size(); ++i)
                        schedule[i] = counts[i];
                    counter[0] = 0;
                    step[0] = 0;
                    for (std::size_t i = 0; i < K_STEPS; ++i)
                        history[i] = 0;
                }

                /** @brief Returns a count the kernels read from counter[0] on the device. */
                [[nodiscard]] LateCount device_count(std::size_t capacity) const
                {
                    return LateCount::from_device(capacity, counter);
                }

                /** @brief Adds the kernel that advances the step and writes its live count. */
                void add_producer(Graph& graph)
                {
                    std::uint32_t* c = counter.data();
                    std::uint32_t* st = step.data();
                    const std::uint32_t* sc = schedule.data();
                    const auto period = static_cast<std::uint32_t>(schedule.size());
                    graph.add(std::size_t{1}, Reads(schedule), Writes(counter, step),
                              [c, st, sc, period](std::size_t)
                              {
                                  const std::uint32_t s = st[0];
                                  c[0] = sc[s % period];
                                  st[0] = s + 1u;
                              });
                }

                /** @brief Adds the kernel that folds words of source into the step's digest. */
                void add_digest(Graph& graph, const Buffer<std::uint32_t>& source,
                                std::size_t words)
                {
                    const std::uint32_t* src = source.data();
                    const std::uint32_t* st = step.data();
                    std::uint32_t* h = history.data();
                    graph.add(std::size_t{1}, Reads(source, step), Writes(history),
                              [src, st, h, words](std::size_t)
                              {
                                  std::uint32_t digest = K_DIGEST_BASIS;
                                  for (std::size_t i = 0; i < words; ++i)
                                      digest = (digest ^ src[i]) * K_DIGEST_PRIME;
                                  h[st[0] - 1u] = digest;
                              });
                }

                /**
                 * @brief Adds the kernel that folds the live words of source into the digest.
                 * @param extra Words the primitive writes past the live count.
                 * @pre source holds stale words above the live count.
                 */
                void add_live_digest(Graph& graph, const Buffer<std::uint32_t>& source,
                                     std::size_t extra)
                {
                    const std::uint32_t* src = source.data();
                    const std::uint32_t* st = step.data();
                    const std::uint32_t* c = counter.data();
                    std::uint32_t* h = history.data();
                    graph.add(std::size_t{1}, Reads(source, step, counter), Writes(history),
                              [src, st, c, h, extra](std::size_t)
                              {
                                  const std::size_t words = static_cast<std::size_t>(c[0]) + extra;
                                  std::uint32_t digest = K_DIGEST_BASIS;
                                  for (std::size_t i = 0; i < words; ++i)
                                      digest = (digest ^ src[i]) * K_DIGEST_PRIME;
                                  h[st[0] - 1u] = digest;
                              });
                }

                /** @brief Runs graph for K_STEPS steps and copies the digests out. */
                Recording record(Graph& graph)
                {
                    Recording recording;
                    const auto report = graph.run(K_STEPS).value();
                    EXPECT_TRUE(report.is_successful);
                    recording.layer = report.layer;
                    recording.digests.assign(history.data(), history.data() + K_STEPS);
                    return recording;
                }
            };

            /**
             * @brief Runs workload on a default and a forced-scheduler runtime and compares them.
             * @param workload Builds its graph on the runtime given and returns its recording.
             */
            void expect_chain_matches_scheduler(const std::function<Recording(Runtime&)>& workload)
            {
                auto chained_runtime = Runtime::create().value();
                SushiRuntime::RuntimeConfig forced;
                forced.force_scheduler = true;
                auto scheduled_runtime = Runtime::create(forced).value();

                const Recording chained = workload(chained_runtime);
                const Recording scheduled = workload(scheduled_runtime);

                EXPECT_EQ(chained.layer, ExecutionLayer::Chain)
                    << "a plan of device-sized primitives and kernels must take the chain";
                EXPECT_EQ(scheduled.layer, ExecutionLayer::Scheduler);
                ASSERT_EQ(chained.digests.size(), scheduled.digests.size());
                EXPECT_EQ(std::memcmp(chained.digests.data(), scheduled.digests.data(),
                                      chained.digests.size() * sizeof(std::uint32_t)), 0);

                std::size_t changes = 0;
                for (std::size_t i = 1; i < chained.digests.size(); ++i)
                    changes += (chained.digests[i] != chained.digests[i - 1]) ? 1u : 0u;
                EXPECT_GT(changes, K_STEPS / 4)
                    << "the live count must change the results step to step";
            }
        } // namespace

        /** @brief Tests reduce() with a device count on both layers as the count changes. */
        TEST(Unit_GraphPrimitivesDeviceSized, ReduceChainMatchesSchedulerAsTheCountChanges)
        {
            constexpr std::size_t K_CAPACITY = 70000;
            expect_chain_matches_scheduler([](Runtime& runtime)
            {
                Harness harness(runtime, K_CAPACITY);
                auto values = runtime.buffer<float>(K_CAPACITY).value();
                auto result = runtime.buffer<float>(1).value();
                auto bits = runtime.buffer<std::uint32_t>(1).value();
                for (std::size_t i = 0; i < K_CAPACITY; ++i)
                    values[i] = static_cast<float>((i * 7919u) % 1013u) * 0.001f + 0.1f;

                auto graph = runtime.graph();
                harness.add_producer(graph);
                GraphPrimitives::reduce(graph.recorder(), values, result,
                                        harness.device_count(K_CAPACITY), Sum<float>{}).value();
                const float* r = result.data();
                std::uint32_t* b = bits.data();
                // The float result reaches the digest as its bit pattern.
                graph.add(std::size_t{1}, Reads(result), Writes(bits),
                          [r, b](std::size_t)
                          {
                              std::uint32_t word = 0;
                              std::memcpy(&word, r, sizeof(word));
                              b[0] = word;
                          });
                harness.add_digest(graph, bits, 1);
                return harness.record(graph);
            });
        }

        /** @brief Tests segmented_reduce() with a device count on both layers. */
        TEST(Unit_GraphPrimitivesDeviceSized, SegmentedReduceChainMatchesScheduler)
        {
            constexpr std::size_t K_SEGMENTS = 500;
            expect_chain_matches_scheduler([](Runtime& runtime)
            {
                Harness harness(runtime, K_SEGMENTS);
                auto offsets = runtime.buffer<std::uint32_t>(K_SEGMENTS + 1).value();
                std::uint32_t at = 0;
                for (std::size_t s = 0; s <= K_SEGMENTS; ++s)
                {
                    offsets[s] = at;
                    at += static_cast<std::uint32_t>(s % 5);
                }
                auto values = runtime.buffer<std::uint32_t>(at).value();
                for (std::size_t i = 0; i < at; ++i)
                    values[i] = static_cast<std::uint32_t>(i) * 2654435761u;
                auto out = runtime.buffer<std::uint32_t>(K_SEGMENTS).value();
                for (std::size_t s = 0; s < K_SEGMENTS; ++s)
                    out[s] = 0;

                auto graph = runtime.graph();
                harness.add_producer(graph);
                GraphPrimitives::segmented_reduce(graph.recorder(), values, offsets, out,
                                                  harness.device_count(K_SEGMENTS),
                                                  Sum<std::uint32_t>{}).value();
                // A segment past the live count keeps an earlier fold, so later steps differ.
                std::uint32_t* o = out.data();
                graph.add(K_SEGMENTS, Reads(out), Writes(out),
                          [o](std::size_t s)
                          {
                              o[s] = o[s] * 3u + static_cast<std::uint32_t>(s);
                          });
                harness.add_digest(graph, out, K_SEGMENTS);
                return harness.record(graph);
            });
        }

        /** @brief Tests prefix_scan() with a device count on both layers. */
        TEST(Unit_GraphPrimitivesDeviceSized, PrefixScanChainMatchesScheduler)
        {
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

        /** @brief Tests counting_sort() with a device count and a host key range on both layers. */
        TEST(Unit_GraphPrimitivesDeviceSized, CountingSortChainMatchesScheduler)
        {
            constexpr std::size_t K_COUNT_CAPACITY = 1000;
            constexpr std::size_t K_KEY_CAPACITY = std::size_t{1} << 17;
            constexpr std::size_t K_BUCKETS = 32;
            constexpr std::size_t K_PACKED = K_COUNT_CAPACITY + K_BUCKETS + 2;
            expect_chain_matches_scheduler([](Runtime& runtime)
            {
                Harness harness(runtime, K_COUNT_CAPACITY);
                auto keys = runtime.buffer<std::uint32_t>(K_COUNT_CAPACITY).value();
                auto indices = runtime.buffer<std::uint32_t>(K_COUNT_CAPACITY).value();
                auto offsets = runtime.buffer<std::uint32_t>(K_KEY_CAPACITY + 1).value();
                auto total = runtime.buffer<std::uint32_t>(1).value();
                auto packed = runtime.buffer<std::uint32_t>(K_PACKED).value();
                for (std::size_t i = 0; i < K_COUNT_CAPACITY; ++i)
                {
                    keys[i] = static_cast<std::uint32_t>((i * 7u + i / 13u) % K_BUCKETS);
                    indices[i] = 0;
                }

                // The live key range changes every step, so one, two and three digits all run.
                const std::array<std::size_t, 4> ranges{K_BUCKETS, 300, 70000, K_KEY_CAPACITY};
                auto tick = std::make_shared<std::size_t>(0);
                const LateCount key_range = LateCount::from_host(
                    K_KEY_CAPACITY, [ranges, tick] { return ranges[(*tick)++ % ranges.size()]; });

                auto graph = runtime.graph();
                harness.add_producer(graph);
                GraphPrimitives::counting_sort(graph.recorder(), keys, indices, offsets, total,
                                               harness.device_count(K_COUNT_CAPACITY), key_range)
                    .value();

                const std::uint32_t* ip = indices.data();
                const std::uint32_t* op = offsets.data();
                const std::uint32_t* tp = total.data();
                std::uint32_t* pp = packed.data();
                graph.add(K_PACKED, Reads(indices, offsets, total), Writes(packed),
                          [ip, op, tp, pp](std::size_t i)
                          {
                              if (i < K_COUNT_CAPACITY)
                                  pp[i] = ip[i];
                              else if (i < K_COUNT_CAPACITY + K_BUCKETS + 1)
                                  pp[i] = op[i - K_COUNT_CAPACITY];
                              else
                                  pp[i] = tp[0];
                          });
                harness.add_digest(graph, packed, K_PACKED);
                return harness.record(graph);
            });
        }

        /** @brief Tests that a device count above capacity fails a chained run of a primitive. */
        TEST(Unit_GraphPrimitivesDeviceSized, DeviceCountOverCapacityFailsAChainedRun)
        {
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

    } // namespace Test
} // namespace SushiBLAS
