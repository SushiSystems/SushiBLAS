/**************************************************************************/
/* device_sized_harness.hpp                                               */
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
 * @file device_sized_harness.hpp
 * @brief Declares the harness that runs a device-sized primitive on the chain and the scheduler.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/graph/graph.hpp>
#include <SushiRuntime/api/runtime/runtime.hpp>
#include <SushiRuntime/api/runtime/runtime_config.hpp>
#include <SushiRuntime/api/vocabulary/dependencies.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>
#include <SushiRuntime/report/run_report.hpp>
#include <SushiBLAS/graph_primitives/combiners.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds the harness the device-sized primitive tests share. */
        namespace DeviceSized
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
            inline constexpr std::size_t K_STEPS = 48;

            /// @brief Holds the offset basis of the 32-bit FNV-1a digest.
            inline constexpr std::uint32_t K_DIGEST_BASIS = 2166136261u;

            /// @brief Holds the prime of the 32-bit FNV-1a digest.
            inline constexpr std::uint32_t K_DIGEST_PRIME = 16777619u;

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
            inline std::vector<std::uint32_t> schedule_for(std::size_t capacity)
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
            inline void
            expect_chain_matches_scheduler(const std::function<Recording(Runtime&)>& workload)
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
        } // namespace DeviceSized
    } // namespace Test
} // namespace SushiBLAS
