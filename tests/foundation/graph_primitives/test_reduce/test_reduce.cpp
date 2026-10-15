/**************************************************************************/
/* test_reduce.cpp                                                        */
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
 * @file test_reduce.cpp
 * @brief Tests reduce(): its fixed combination order, its late-bound count and its node widths.
 * @author Mustafa Garip
 * @see modules/foundation/graph_primitives/tests/README.md
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

#include <gtest/gtest.h>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/graph/graph.hpp>
#include <SushiRuntime/api/runtime/runtime.hpp>
#include <SushiRuntime/api/vocabulary/dependencies.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>
#include <SushiRuntime/core/error.hpp>

#include <SushiBLAS/graph_primitives/combiners.hpp>
#include <SushiBLAS/graph_primitives/fixed_order_reduce.hpp>
#include <SushiBLAS/graph_primitives/reduce.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using GraphPrimitives::Maximum;
            using GraphPrimitives::Minimum;
            using GraphPrimitives::Sum;
            using GraphPrimitives::reduce;
            using SushiRuntime::LateCount;
            using SushiRuntime::Reads;
            using SushiRuntime::Runtime;
            using SushiRuntime::Writes;

            /// @brief Holds the tile size the host reference folds with.
            constexpr std::size_t K_TILE = GraphPrimitives::K_REDUCE_TILE;

            /** @brief Folds each tile of in left to right and returns one partial per tile. */
            template <typename T>
            std::vector<T> fold_level(const std::vector<T>& in)
            {
                std::vector<T> out;
                for (std::size_t begin = 0; begin < in.size(); begin += K_TILE)
                {
                    const std::size_t end =
                        (begin + K_TILE < in.size()) ? begin + K_TILE : in.size();
                    T accumulator = in[begin];
                    for (std::size_t j = begin + 1; j < end; ++j)
                        accumulator = accumulator + in[j];
                    out.push_back(accumulator);
                }
                return out;
            }

            /** @brief Returns the sum reduce() must produce, by folding levels down to one. */
            template <typename T>
            T fixed_order_sum(std::vector<T> values)
            {
                while (values.size() > 1)
                    values = fold_level(values);
                return values.front();
            }

            /** @brief Returns one large value and n - 1 ones, whose sum depends on the order. */
            std::vector<float> order_sensitive_input(std::size_t n)
            {
                std::vector<float> values(n, 1.0f);
                values[0] = 1.0e8f;
                return values;
            }

            /** @brief Returns whether two floats have the same bit pattern. */
            bool bit_equal(float a, float b)
            {
                std::uint32_t pattern_a = 0;
                std::uint32_t pattern_b = 0;
                std::memcpy(&pattern_a, &a, sizeof(pattern_a));
                std::memcpy(&pattern_b, &b, sizeof(pattern_b));
                return pattern_a == pattern_b;
            }

            /** @brief Returns whether two doubles have the same bit pattern. */
            bool bit_equal(double a, double b)
            {
                std::uint64_t pattern_a = 0;
                std::uint64_t pattern_b = 0;
                std::memcpy(&pattern_a, &a, sizeof(pattern_a));
                std::memcpy(&pattern_b, &b, sizeof(pattern_b));
                return pattern_a == pattern_b;
            }

            /** @brief Expects a record call to have returned an InvalidGraph error. */
            void expect_invalid_graph(const SushiRuntime::Result<void>& recorded)
            {
                ASSERT_FALSE(recorded.has_value()) << "the call recorded instead of refusing";
                EXPECT_EQ(recorded.error().code, SushiRuntime::ErrorCode::InvalidGraph);
            }
        } // namespace

        /** @brief Tests that the sum equals the fixed-order host fold and not the naive sum. */
        TEST(Unit_GraphPrimitivesReduce, SumMatchesTheFixedOrderReferenceExactly)
        {
            constexpr std::size_t K_COUNT = 100000;

            auto runtime = Runtime::create().value();
            auto in = runtime.buffer<float>(K_COUNT).value();
            auto out = runtime.buffer<float>(1).value();

            const std::vector<float> host = order_sensitive_input(K_COUNT);
            for (std::size_t i = 0; i < K_COUNT; ++i)
                in[i] = host[i];
            out[0] = -1.0f;

            auto graph = runtime.graph();
            reduce(graph.recorder(), in, out, K_COUNT, Sum<float>{}).value();
            graph.run().value();

            const float expected = fixed_order_sum(host);
            EXPECT_TRUE(bit_equal(out[0], expected))
                << "got " << out[0] << ", fixed-order reference " << expected;

            float naive = host[0];
            for (std::size_t i = 1; i < K_COUNT; ++i)
                naive = naive + host[i];
            EXPECT_FALSE(bit_equal(naive, expected))
                << "the input is not order-sensitive, so this test cannot detect a "
                   "schedule-dependent combination order";
        }

        /** @brief Tests that repeated runs and a run with the rebalancer on agree bit for bit. */
        TEST(Unit_GraphPrimitivesReduce, ResultIsStableAcrossSchedules)
        {
            constexpr std::size_t K_COUNT = 70000;
            constexpr int K_ROUNDS = 32;

            const std::vector<float> host = order_sensitive_input(K_COUNT);

            auto runtime = Runtime::create().value();
            auto in = runtime.buffer<float>(K_COUNT).value();
            auto out = runtime.buffer<float>(1).value();
            for (std::size_t i = 0; i < K_COUNT; ++i)
                in[i] = host[i];

            auto graph = runtime.graph();
            reduce(graph.recorder(), in, out, K_COUNT, Sum<float>{}).value();

            graph.run().value();
            const float first = out[0];

            for (int round = 1; round < K_ROUNDS; ++round)
            {
                out[0] = 0.0f;
                graph.run().value();
                ASSERT_TRUE(bit_equal(out[0], first)) << "round " << round;
            }
            EXPECT_EQ(graph.compile_count(), 1u) << "a replayed reduction must not recompile";

            // With the rebalancer on, task migration reassigns tiles to workers during a run.
            auto rebalanced = Runtime::create().value();
            rebalanced.set_rebalancer_enabled(true);
            auto rebalanced_in = rebalanced.buffer<float>(K_COUNT).value();
            auto rebalanced_out = rebalanced.buffer<float>(1).value();
            for (std::size_t i = 0; i < K_COUNT; ++i)
                rebalanced_in[i] = host[i];

            auto rebalanced_graph = rebalanced.graph();
            reduce(rebalanced_graph.recorder(), rebalanced_in, rebalanced_out, K_COUNT,
                   Sum<float>{}).value();
            rebalanced_graph.run().value();

            EXPECT_TRUE(bit_equal(rebalanced_out[0], first))
                << "the rebalancer changed the result: the combination order is not fixed";
        }

        /** @brief Tests counts at the tile boundaries, a single value and an empty reduction. */
        TEST(Unit_GraphPrimitivesReduce, EdgeCounts)
        {
            const std::size_t counts[] = {1, 2, K_TILE - 1, K_TILE, K_TILE + 1,
                                          K_TILE * K_TILE, K_TILE * K_TILE + 1};

            auto runtime = Runtime::create().value();

            for (std::size_t n : counts)
            {
                auto in = runtime.buffer<double>(n).value();
                auto out = runtime.buffer<double>(1).value();
                std::vector<double> host(n);
                for (std::size_t i = 0; i < n; ++i)
                {
                    host[i] = static_cast<double>(i % 7) + 0.5;
                    in[i] = host[i];
                }
                out[0] = -1.0;

                auto graph = runtime.graph();
                reduce(graph.recorder(), in, out, n, Sum<double>{}).value();
                graph.run().value();

                EXPECT_TRUE(bit_equal(out[0], fixed_order_sum(host))) << "n = " << n;
            }

            // A count of zero produces the identity without reading the input.
            auto in = runtime.buffer<double>(4).value();
            auto out = runtime.buffer<double>(1).value();
            out[0] = 123.0;
            auto graph = runtime.graph();
            reduce(graph.recorder(), in, out, std::size_t{0}, Sum<double>{}).value();
            graph.run().value();
            EXPECT_TRUE(bit_equal(out[0], 0.0));
        }

        /** @brief Tests the Minimum and Maximum reductions against a host scan. */
        TEST(Unit_GraphPrimitivesReduce, MinimumAndMaximum)
        {
            constexpr std::size_t K_COUNT = 5000;

            auto runtime = Runtime::create().value();
            auto in = runtime.buffer<int>(K_COUNT).value();
            auto lowest = runtime.buffer<int>(1).value();
            auto highest = runtime.buffer<int>(1).value();

            for (std::size_t i = 0; i < K_COUNT; ++i)
                in[i] = static_cast<int>((i * 7919) % 10007) - 5000;

            int expected_lowest = in[0];
            int expected_highest = in[0];
            for (std::size_t i = 1; i < K_COUNT; ++i)
            {
                if (in[i] < expected_lowest)
                    expected_lowest = in[i];
                if (in[i] > expected_highest)
                    expected_highest = in[i];
            }

            auto graph = runtime.graph();
            reduce(graph.recorder(), in, lowest, K_COUNT, Minimum<int>{}).value();
            reduce(graph.recorder(), in, highest, K_COUNT, Maximum<int>{}).value();
            graph.run().value();

            EXPECT_EQ(lowest[0], expected_lowest);
            EXPECT_EQ(highest[0], expected_highest);
        }

        /** @brief Tests that a reader of the output is ordered after the whole level chain. */
        TEST(Unit_GraphPrimitivesReduce, ConsumerSeesTheFinishedResult)
        {
            constexpr std::size_t K_COUNT = 40000;

            auto runtime = Runtime::create().value();
            auto in = runtime.buffer<int>(K_COUNT).value();
            auto total = runtime.buffer<int>(1).value();
            auto twice = runtime.buffer<int>(1).value();

            for (std::size_t i = 0; i < K_COUNT; ++i)
                in[i] = 1;
            total[0] = 0;
            twice[0] = 0;

            const int* total_base = total.data();
            int* twice_base = twice.data();

            auto graph = runtime.graph();
            reduce(graph.recorder(), in, total, K_COUNT, Sum<int>{}).value();
            graph.add(std::size_t{1}, Reads(total), Writes(twice),
                      [total_base, twice_base](std::size_t)
                      {
                          twice_base[0] = total_base[0] * 2;
                      });
            graph.run().value();

            EXPECT_EQ(total[0], static_cast<int>(K_COUNT));
            EXPECT_EQ(twice[0], static_cast<int>(K_COUNT) * 2);
        }

        /** @brief Tests that a mis-sized input or output is refused when the call records. */
        TEST(Unit_GraphPrimitivesReduce, RejectsMisSizedArguments)
        {
            auto runtime = Runtime::create().value();
            auto in = runtime.buffer<float>(16).value();
            auto out = runtime.buffer<float>(1).value();
            auto empty = runtime.buffer<float>(0).value();

            auto graph = runtime.graph();
            expect_invalid_graph(reduce(graph.recorder(), in, out, std::size_t{17}, Sum<float>{}));
            expect_invalid_graph(reduce(graph.recorder(), in, empty, std::size_t{4}, Sum<float>{}));
            expect_invalid_graph(
                reduce(graph.recorder(), in, out, LateCount::fixed(17), Sum<float>{}));
            expect_invalid_graph(
                reduce(graph.recorder(), in, empty, LateCount::fixed(4), Sum<float>{}));
            EXPECT_EQ(graph.recorder().node_count(), 0u) << "a refused call must record nothing";
        }

        /** @brief Tests that a late-bound reduce equals the fixed-count one at each live count. */
        TEST(Unit_GraphPrimitivesReduce, LateBoundMatchesFixedCountExactly)
        {
            constexpr std::size_t K_CAPACITY = K_TILE * K_TILE + 1;
            const std::size_t counts[] = {1, K_TILE - 1, K_TILE, K_TILE + 1, K_CAPACITY};

            auto runtime = Runtime::create().value();

            for (std::size_t n : counts)
            {
                auto in = runtime.buffer<double>(K_CAPACITY).value();
                auto out_fixed = runtime.buffer<double>(1).value();
                auto out_late = runtime.buffer<double>(1).value();

                for (std::size_t i = 0; i < n; ++i)
                    in[i] = static_cast<double>(i % 7) + 0.5;
                out_fixed[0] = -1.0;
                out_late[0] = -1.0;

                auto graph = runtime.graph();
                reduce(graph.recorder(), in, out_fixed, n, Sum<double>{}).value();
                reduce(graph.recorder(), in, out_late,
                       LateCount::from_host(K_CAPACITY, [n]() { return n; }), Sum<double>{})
                    .value();
                graph.run().value();

                EXPECT_TRUE(bit_equal(out_late[0], out_fixed[0]))
                    << "n = " << n << ": late " << out_late[0] << ", fixed " << out_fixed[0];
            }
        }

        /** @brief Tests that a live count that changes between runs keeps one compile. */
        TEST(Unit_GraphPrimitivesReduce, LateBoundChangingCountKeepsOneCompile)
        {
            constexpr std::size_t K_CAPACITY = 700000;

            auto runtime = Runtime::create().value();
            auto in = runtime.buffer<double>(K_CAPACITY).value();
            auto out = runtime.buffer<double>(1).value();
            for (std::size_t i = 0; i < K_CAPACITY; ++i)
                in[i] = 1.0;

            std::size_t live_n = 1000;
            const LateCount count =
                LateCount::from_host(K_CAPACITY, [&live_n]() { return live_n; });

            auto graph = runtime.graph();
            reduce(graph.recorder(), in, out, count, Sum<double>{}).value();

            // Sums of ones below 2^53 are exact, so the comparisons below are exact.
            graph.run().value();
            EXPECT_EQ(out[0], 1000.0);

            live_n = 700000;
            graph.run().value();
            EXPECT_EQ(out[0], 700000.0);

            live_n = 3;
            graph.run().value();
            EXPECT_EQ(out[0], 3.0);

            live_n = 0;
            graph.run().value();
            EXPECT_EQ(out[0], 0.0) << "a live count of zero must write the identity";

            EXPECT_EQ(graph.compile_count(), 1u)
                << "a live count change between runs must not trigger a recompile";
        }

        /** @brief Tests that a live count above its capacity fails the run. */
        TEST(Unit_GraphPrimitivesReduce, LateBoundAboveCapacityFailsTheRun)
        {
            auto runtime = Runtime::create().value();
            auto in = runtime.buffer<double>(100).value();
            auto out = runtime.buffer<double>(1).value();

            auto graph = runtime.graph();
            reduce(graph.recorder(), in, out,
                   LateCount::from_host(100, []() { return std::size_t{101}; }), Sum<double>{})
                .value();
            EXPECT_FALSE(graph.run().has_value());
        }

        /** @brief Tests that a late-bound reduce folds the count its predecessor wrote that run. */
        TEST(Unit_GraphPrimitivesReduce, DeviceCountIsReadAfterItsProducer)
        {
            constexpr std::size_t K_CAPACITY = 1000;
            using SushiRuntime::DeviceIndex;
            using SushiRuntime::Residency;

            auto runtime = Runtime::create().value();
            auto values = runtime.buffer<double>(K_CAPACITY).value();
            auto counter =
                runtime.buffer<std::uint32_t>(1, DeviceIndex{0}, Residency::Host).value();
            auto source = runtime.buffer<std::uint32_t>(1, DeviceIndex{0}, Residency::Host).value();
            auto out = runtime.buffer<double>(1).value();

            for (std::size_t i = 0; i < K_CAPACITY; ++i)
                values[i] = 1.0;

            // A device kernel cannot reach this frame, so the next count sits in a buffer.
            const std::uint32_t* source_base = source.data();
            std::uint32_t* counter_base = counter.data();

            auto graph = runtime.graph();
            graph.add(std::size_t{1}, Reads(source), Writes(counter),
                      [counter_base, source_base](std::size_t)
                      {
                          counter_base[0] = source_base[0];
                      });
            reduce(graph.recorder(), values, out, LateCount::from_device(K_CAPACITY, counter),
                   Sum<double>{}).value();

            source[0] = 300;
            graph.run().value();
            EXPECT_EQ(out[0], 300.0) << "the reduce must fold the count this run's producer wrote";

            source[0] = 700;
            graph.run().value();
            EXPECT_EQ(out[0], 700.0) << "a changed count must take effect the same run";

            EXPECT_EQ(graph.compile_count(), 1u);
        }

        /** @brief Tests that each level node launches at the out_count of its level. */
        TEST(Unit_GraphPrimitivesReduce, LevelWidthsMatchTheLevelPlan)
        {
            constexpr std::size_t K_COUNT = 2 * K_TILE + 88;
            const std::vector<std::uint32_t> expected_widths = {3u, 1u};

            auto runtime = Runtime::create().value();
            auto values = runtime.buffer<float>(K_COUNT).value();
            auto total = runtime.buffer<float>(1).value();
            for (std::size_t i = 0; i < K_COUNT; ++i)
                values[i] = 1.0f;

            auto graph = runtime.graph();
            graph.set_profiling(true);
            reduce(graph.recorder(), values, total, K_COUNT, Sum<float>{}).value();
            EXPECT_EQ(graph.recorder().rule_count(), 1u) << "the levels of one call are one rule";

            const auto report = graph.run().value();

            ASSERT_EQ(report.node_events.size(), expected_widths.size());
            for (std::size_t i = 0; i < expected_widths.size(); ++i)
            {
                EXPECT_EQ(report.node_events[i].launched_width, expected_widths[i])
                    << "level " << i;
                EXPECT_EQ(report.node_events[i].live_width, expected_widths[i]) << "level " << i;
            }
            EXPECT_EQ(total[0], static_cast<float>(K_COUNT));
        }

        /** @brief Tests that a fixed count of zero records one node of width one. */
        TEST(Unit_GraphPrimitivesReduce, ZeroCountStampsWidthOne)
        {
            auto runtime = Runtime::create().value();
            auto values = runtime.buffer<float>(1).value();
            auto total = runtime.buffer<float>(1).value();
            total[0] = -1.0f;

            auto graph = runtime.graph();
            graph.set_profiling(true);
            reduce(graph.recorder(), values, total, std::size_t{0}, Sum<float>{}).value();

            const auto report = graph.run().value();

            ASSERT_EQ(report.node_events.size(), 1u);
            EXPECT_EQ(report.node_events[0].launched_width, 1u);
            EXPECT_EQ(report.node_events[0].live_width, 1u);
            EXPECT_EQ(total[0], 0.0f);
        }

    } // namespace Test
} // namespace SushiBLAS
