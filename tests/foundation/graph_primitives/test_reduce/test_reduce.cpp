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
 * @file test_reduce.cpp
 * @brief Tests reduce(): its fixed combination order, its late-bound count and its node widths.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/graph/graph.hpp>
#include <SushiRuntime/api/runtime/runtime.hpp>
#include <SushiRuntime/api/vocabulary/dependencies.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>
#include <SushiRuntime/core/determinism.hpp>
#include <SushiRuntime/core/error.hpp>

#include <SushiBLAS/graph_primitives/combiners.hpp>
#include <SushiBLAS/graph_primitives/fixed_order_reduce.hpp>
#include <SushiBLAS/graph_primitives/fixed_order_scan.hpp>
#include <SushiBLAS/graph_primitives/host_primitive_recorder.hpp>
#include <SushiBLAS/graph_primitives/primitive_options.hpp>
#include <SushiBLAS/graph_primitives/reduce.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

#include "../host_nodes.hpp"
#include "../plan_labels.hpp"
#include "../window_probe.hpp"
#include "../runtime_port_calls.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using GraphPrimitives::Maximum;
            using GraphPrimitives::Minimum;
            using GraphPrimitives::PrimitiveOptions;
            using GraphPrimitives::ReduceWindow;
            using GraphPrimitives::Sum;
            using RuntimePort::reduce;
            using SushiRuntime::Buffer;
            using SushiRuntime::Determinism;
            using SushiRuntime::LateCount;
            using SushiRuntime::Reads;
            using SushiRuntime::Runtime;
            using SushiRuntime::Writes;

            /// @brief Holds the tile size the host reference folds with.
            constexpr std::size_t K_TILE = GraphPrimitives::K_REDUCE_TILE;

            /// @brief Holds the capacity of the late-bound rows: two levels and one value more.
            constexpr std::size_t K_LATE_CAPACITY = K_TILE * K_TILE + 1;

            /// @brief Holds the capacity of the changing-count graph.
            constexpr std::size_t K_CHANGING_CAPACITY = 700000;

            /// @brief Holds the live count the changing-count graph runs with first.
            constexpr std::size_t K_FIRST_LIVE_COUNT = 1000;

            /// @brief Holds a count that folds in two levels, three tiles wide and then one.
            constexpr std::size_t K_TWO_LEVEL_COUNT = 2 * K_TILE + 88;

            /// @brief Holds the count of the window that the ordering probes run against.
            constexpr std::size_t K_PROBED_COUNT = K_TILE + 3;

            /// @brief Holds the element of the input the probed window starts at.
            constexpr std::size_t K_PROBED_OFFSET = 40;

            /// @brief Holds the elements of the input past the probed window.
            constexpr std::size_t K_PROBED_TAIL = 8;

            /// @brief Holds the count of the window that the metamorphic rows reduce.
            constexpr std::size_t K_WINDOW_COUNT = 2 * K_TILE + 11;

            /// @brief Holds the element of the wide input the window starts at.
            constexpr std::size_t K_WINDOW_INPUT_OFFSET = 37;

            /// @brief Holds the elements of the wide input past the window.
            constexpr std::size_t K_WINDOW_INPUT_TAIL = 5;

            /// @brief Holds the element of the wide output the window writes.
            constexpr std::size_t K_WINDOW_OUTPUT_OFFSET = 2;

            /// @brief Holds the size of the wide output.
            constexpr std::size_t K_WINDOW_OUTPUT_SIZE = K_WINDOW_OUTPUT_OFFSET + 3;

            /// @brief Holds the value the wide input holds outside the window.
            constexpr float K_INPUT_CANARY = 1.0e6f;

            /// @brief Holds the value the wide output holds outside the window.
            constexpr float K_OUTPUT_CANARY = -7.0f;

            /// @brief Names how a row passes its count to reduce().
            enum class CountForm
            {
                Fixed,
                Late
            };

            /// @brief Names the operand a probe writes a part of.
            enum class Operand
            {
                Input,
                Output
            };

            /// @brief Names a part of an operand a probe writes, and whether the reduce orders it.
            struct ProbeRow
            {
                /// @brief Names the row in the test report.
                const char* name;
                /// @brief Holds the operand whose buffer the probe writes.
                Operand operand;
                /// @brief Holds the first element the probe writes.
                std::size_t offset;
                /// @brief Holds how many elements the probe writes.
                std::size_t count;
                /// @brief Holds whether the probe must wait for the reduce.
                bool ordered;
            };

            /// @brief Names a shape of a named call and the node names it must record.
            struct NamedShapeRow
            {
                /// @brief Names the row in the test report.
                const char* name;
                /// @brief Holds how the call passes its count.
                CountForm form;
                /// @brief Holds the count the call reduces.
                std::size_t count;
                /// @brief Holds the name of each node, in recording order.
                std::vector<std::string> names;
            };

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

            /** @brief Returns the sum of values added one at a time from the first. */
            float left_to_right_sum(const std::vector<float>& values)
            {
                float sum = values.front();
                for (std::size_t i = 1; i < values.size(); ++i)
                    sum = sum + values[i];
                return sum;
            }

            /** @brief Returns one large value and n - 1 ones, whose sum depends on the order. */
            std::vector<float> order_sensitive_input(std::size_t n)
            {
                std::vector<float> values(n, 1.0f);
                values[0] = 1.0e8f;
                return values;
            }

            /** @brief Returns n doubles 0.5, 1.5, ..., 6.5 repeated, each exact in binary. */
            std::vector<double> repeating_halves(std::size_t n)
            {
                std::vector<double> values;
                values.reserve(n);
                for (std::size_t i = 0; i < n; ++i)
                    values.push_back(static_cast<double>(i % 7) + 0.5);
                return values;
            }

            /** @brief Returns n distinct integers from -5000 to 5006 in a scattered order. */
            std::vector<int> scattered_integers(std::size_t n)
            {
                std::vector<int> values;
                values.reserve(n);
                for (std::size_t i = 0; i < n; ++i)
                    values.push_back(static_cast<int>((i * 7919) % 10007) - 5000);
                return values;
            }

            /** @brief Returns values with before canaries ahead of it and after canaries behind. */
            std::vector<float> framed(const std::vector<float>& values, std::size_t before,
                                      std::size_t after, float canary)
            {
                std::vector<float> frame(before, canary);
                frame.insert(frame.end(), values.begin(), values.end());
                frame.insert(frame.end(), after, canary);
                return frame;
            }

            /** @brief Returns a buffer of runtime that holds values, element for element. */
            template <typename T>
            Buffer<T> buffer_holding(Runtime& runtime, const std::vector<T>& values)
            {
                Buffer<T> buffer = runtime.buffer<T>(values.size()).value();
                for (std::size_t i = 0; i < values.size(); ++i)
                    buffer[i] = values[i];
                return buffer;
            }

            /** @brief Returns every element of buffer but the one at skipped, in order. */
            std::vector<float> elements_except(Buffer<float>& buffer, std::size_t skipped)
            {
                std::vector<float> elements;
                for (std::size_t i = 0; i < buffer.size(); ++i)
                {
                    if (i != skipped)
                        elements.push_back(buffer[i]);
                }
                return elements;
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

            /** @brief Runs graph rounds times and returns how many runs left first in out[0]. */
            int runs_matching(SushiRuntime::Graph& graph, Buffer<float>& out, float first,
                              int rounds)
            {
                int matching = 0;
                for (int round = 0; round < rounds; ++round)
                {
                    out[0] = 0.0f;
                    graph.run().value();
                    if (bit_equal(out[0], first))
                        ++matching;
                }
                return matching;
            }

            /** @brief Records a Sum reduce of count values, passing the count in form. */
            template <typename T>
            SushiRuntime::Result<void>
            record_sum(SushiRuntime::Graph& graph, const Buffer<T>& in, const Buffer<T>& out,
                       CountForm form, std::size_t count, ReduceWindow window = {},
                       const std::optional<PrimitiveOptions>& options = std::nullopt)
            {
                if (form == CountForm::Late)
                    return reduce(graph.recorder(), in, out, LateCount::fixed(count), Sum<T>{},
                                  window, options);
                return reduce(graph.recorder(), in, out, count, Sum<T>{}, window, options);
            }

            /// @brief Names the recorder whose operands the host rows use.
            using HostRecorder = GraphPrimitives::HostPrimitiveRecorder;

            /** @brief Records a Sum reduce of count values on a host recorder, passing the count in form. */
            template <typename T>
            SushiRuntime::Result<void>
            record_sum(GraphPrimitives::HostPrimitiveRecorder& recorder,
                       const GraphPrimitives::HostBuffer<T>& in,
                       const GraphPrimitives::HostBuffer<T>& out, CountForm form,
                       std::size_t count,
                       ReduceWindow window = {},
                       const std::optional<PrimitiveOptions>& options = std::nullopt)
            {
                if (form == CountForm::Late)
                    return GraphPrimitives::reduce(recorder, in, out, LateCount::fixed(count),
                                                   Sum<T>{}, window, options);
                return GraphPrimitives::reduce(recorder, in, out, count, Sum<T>{}, window, options);
            }

            /** @brief Returns the part of in or out that row names, as a region. */
            SushiRuntime::BufferRegion probed_region(const ProbeRow& row,
                                                     const GraphPrimitives::HostBuffer<float>& in,
                                                     const GraphPrimitives::HostBuffer<float>& out)
            {
                if (row.operand == Operand::Input)
                    return WindowProbe::region_of(in, row.offset, row.count);
                return WindowProbe::region_of(out, row.offset, row.count);
            }

            /** @brief Returns the options that name every node "sum" and state Bitwise. */
            PrimitiveOptions sum_options()
            {
                return {"sum", Determinism::Bitwise};
            }

            /** @brief Returns the probed parts around the window, and whether each is ordered. */
            std::vector<ProbeRow> probe_rows()
            {
                return {
                    {"InputBeforeTheWindow", Operand::Input, 0, K_PROBED_OFFSET, false},
                    {"InputAfterTheWindow", Operand::Input, K_PROBED_OFFSET + K_PROBED_COUNT,
                     K_PROBED_TAIL, false},
                    {"LastInputElementOfTheWindow", Operand::Input,
                     K_PROBED_OFFSET + K_PROBED_COUNT - 1, 2, true},
                    {"OutputBeforeTheWindow", Operand::Output, 0, 1, false},
                    {"OutputTheWindowNames", Operand::Output, 1, 1, true},
                };
            }

            /** @brief Returns the three shapes of a named call and the names each records. */
            std::vector<NamedShapeRow> named_shape_rows()
            {
                const std::size_t levels =
                    GraphPrimitives::Detail::plan_tiled_levels(K_TWO_LEVEL_COUNT).size();
                std::vector<std::string> late_stages = {"resolve"};
                late_stages.insert(late_stages.end(), levels, "level");
                late_stages.push_back("close");
                return {
                    {"Fixed", CountForm::Fixed, K_TWO_LEVEL_COUNT,
                     std::vector<std::string>(
                         GraphPrimitives::Detail::plan_reduction(K_TWO_LEVEL_COUNT).size(),
                         "sum.level")},
                    {"Late", CountForm::Late, K_TWO_LEVEL_COUNT,
                     PlanLabels::staged("sum", late_stages)},
                    {"Empty", CountForm::Fixed, 0, {"sum.level"}},
                };
            }

            /** @brief Returns the name of a count form. */
            std::string form_name(CountForm form)
            {
                return form == CountForm::Late ? "Late" : "Fixed";
            }

            /** @brief Names a test case after its count form. */
            std::string form_param_name(const ::testing::TestParamInfo<CountForm>& info)
            {
                return form_name(info.param);
            }

            /** @brief Names a test case after its row. */
            template <typename Row>
            std::string row_name(const ::testing::TestParamInfo<Row>& info)
            {
                return info.param.name;
            }

            /** @brief Names a test case after its row and its count form. */
            template <typename Row>
            std::string row_form_name(
                const ::testing::TestParamInfo<std::tuple<Row, CountForm>>& info)
            {
                return std::string(std::get<0>(info.param).name) +
                       form_name(std::get<1>(info.param));
            }
        } // namespace

        /** @brief Tests that the sum equals the fixed-order host fold and not the naive sum. */
        TEST(Medium_GraphPrimitivesReduce, SumMatchesTheFixedOrderReferenceExactly)
        {
            constexpr std::size_t K_COUNT = 100000;
            const std::vector<float> host = order_sensitive_input(K_COUNT);
            const float expected = fixed_order_sum(host);
            ASSERT_FALSE(bit_equal(left_to_right_sum(host), expected))
                << "the input is not order-sensitive, so this test cannot detect a "
                   "schedule-dependent combination order";

            SushiRuntime::Runtime& runtime = SushiBLAS::Test::TestEnvironment::runtime();
            auto in = buffer_holding(runtime, host);
            auto out = buffer_holding(runtime, std::vector<float>{-1.0f});

            auto graph = runtime.graph();
            reduce(graph.recorder(), in, out, K_COUNT, Sum<float>{}).value();
            graph.run().value();

            EXPECT_TRUE(bit_equal(out[0], expected))
                << "got " << out[0] << ", fixed-order reference " << expected;
        }

        /** @brief Tests that repeated runs of one graph agree bit for bit and never recompile. */
        TEST(Medium_GraphPrimitivesReduce, RepeatedRunsAgreeBitForBit)
        {
            constexpr std::size_t K_COUNT = 70000;
            constexpr int K_REPLAYS = 31;

            SushiRuntime::Runtime& runtime = SushiBLAS::Test::TestEnvironment::runtime();
            auto in = buffer_holding(runtime, order_sensitive_input(K_COUNT));
            auto out = runtime.buffer<float>(1).value();

            auto graph = runtime.graph();
            reduce(graph.recorder(), in, out, K_COUNT, Sum<float>{}).value();
            graph.run().value();
            const float first = out[0];

            EXPECT_EQ(runs_matching(graph, out, first, K_REPLAYS), K_REPLAYS);
            EXPECT_EQ(graph.compile_count(), 1u) << "a replayed reduction must not recompile";
        }

        /** @brief Tests that a run with the rebalancer on gives the fixed-order sum. */
        TEST(Medium_GraphPrimitivesReduce, RebalancedRunMatchesTheFixedOrderReference)
        {
            constexpr std::size_t K_COUNT = 70000;
            const std::vector<float> host = order_sensitive_input(K_COUNT);

            // With the rebalancer on, task migration reassigns tiles to workers during a run.
            SushiRuntime::Runtime& runtime = SushiBLAS::Test::TestEnvironment::runtime();
            runtime.set_rebalancer_enabled(true);
            auto in = buffer_holding(runtime, host);
            auto out = runtime.buffer<float>(1).value();

            auto graph = runtime.graph();
            reduce(graph.recorder(), in, out, K_COUNT, Sum<float>{}).value();
            graph.run().value();

            EXPECT_TRUE(bit_equal(out[0], fixed_order_sum(host)))
                << "the rebalancer changed the result: the combination order is not fixed";
        }

        /// @brief Gives a reduce test one element count to run at.
        class ReduceAtCount : public ::testing::TestWithParam<std::size_t>
        {
        };

        /// @brief Runs a fixed-count Sum at one count around the tile boundaries.
        class Medium_GraphPrimitivesReduceEdgeCount : public ReduceAtCount
        {
        };

        /** @brief Tests that a count at a tile boundary gives the fixed-order sum. */
        TEST_P(Medium_GraphPrimitivesReduceEdgeCount, EdgeCounts)
        {
            const std::vector<double> host = repeating_halves(GetParam());

            SushiRuntime::Runtime& runtime = SushiBLAS::Test::TestEnvironment::runtime();
            auto in = buffer_holding(runtime, host);
            auto out = buffer_holding(runtime, std::vector<double>{-1.0});

            auto graph = runtime.graph();
            reduce(graph.recorder(), in, out, GetParam(), Sum<double>{}).value();
            graph.run().value();

            EXPECT_TRUE(bit_equal(out[0], fixed_order_sum(host)));
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_GraphPrimitivesReduceEdgeCount,
                                 ::testing::Values(std::size_t{1}, std::size_t{2}, K_TILE - 1,
                                                   K_TILE, K_TILE + 1, K_TILE * K_TILE,
                                                   K_TILE * K_TILE + 1));

        /** @brief Tests that a count of zero writes the identity without reading the input. */
        TEST(Medium_GraphPrimitivesReduce, ZeroCountWritesTheIdentity)
        {
            SushiRuntime::Runtime& runtime = SushiBLAS::Test::TestEnvironment::runtime();
            auto in = runtime.buffer<double>(4).value();
            auto out = buffer_holding(runtime, std::vector<double>{123.0});

            auto graph = runtime.graph();
            reduce(graph.recorder(), in, out, std::size_t{0}, Sum<double>{}).value();
            graph.run().value();

            EXPECT_TRUE(bit_equal(out[0], 0.0));
        }

        /** @brief Tests that the Minimum reduction gives the smallest value. */
        TEST(Medium_GraphPrimitivesReduce, MinimumGivesTheSmallestValue)
        {
            const std::vector<int> host = scattered_integers(5000);

            SushiRuntime::Runtime& runtime = SushiBLAS::Test::TestEnvironment::runtime();
            auto in = buffer_holding(runtime, host);
            auto lowest = runtime.buffer<int>(1).value();

            auto graph = runtime.graph();
            reduce(graph.recorder(), in, lowest, host.size(), Minimum<int>{}).value();
            graph.run().value();

            EXPECT_EQ(lowest[0], *std::min_element(host.begin(), host.end()));
        }

        /** @brief Tests that the Maximum reduction gives the largest value. */
        TEST(Medium_GraphPrimitivesReduce, MaximumGivesTheLargestValue)
        {
            const std::vector<int> host = scattered_integers(5000);

            SushiRuntime::Runtime& runtime = SushiBLAS::Test::TestEnvironment::runtime();
            auto in = buffer_holding(runtime, host);
            auto highest = runtime.buffer<int>(1).value();

            auto graph = runtime.graph();
            reduce(graph.recorder(), in, highest, host.size(), Maximum<int>{}).value();
            graph.run().value();

            EXPECT_EQ(highest[0], *std::max_element(host.begin(), host.end()));
        }

        /** @brief Tests that a reader of the output is ordered after the whole level chain. */
        TEST(Medium_GraphPrimitivesReduce, ConsumerSeesTheFinishedResult)
        {
            constexpr std::size_t K_COUNT = 40000;

            SushiRuntime::Runtime& runtime = SushiBLAS::Test::TestEnvironment::runtime();
            auto in = buffer_holding(runtime, std::vector<int>(K_COUNT, 1));
            auto total = buffer_holding(runtime, std::vector<int>{0});
            auto twice = buffer_holding(runtime, std::vector<int>{0});

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

        /// @brief Runs a late-bound and a fixed-count Sum at one live count.
        class Medium_GraphPrimitivesReduceLateCount : public ReduceAtCount
        {
        };

        /** @brief Tests that a late-bound reduce equals the fixed-count one at each live count. */
        TEST_P(Medium_GraphPrimitivesReduceLateCount, LateBoundMatchesFixedCountExactly)
        {
            const std::size_t n = GetParam();

            SushiRuntime::Runtime& runtime = SushiBLAS::Test::TestEnvironment::runtime();
            auto in = buffer_holding(runtime, repeating_halves(K_LATE_CAPACITY));
            auto out_fixed = buffer_holding(runtime, std::vector<double>{-1.0});
            auto out_late = buffer_holding(runtime, std::vector<double>{-1.0});

            auto graph = runtime.graph();
            reduce(graph.recorder(), in, out_fixed, n, Sum<double>{}).value();
            reduce(graph.recorder(), in, out_late,
                   LateCount::from_host(K_LATE_CAPACITY, [n]() { return n; }), Sum<double>{})
                .value();
            graph.run().value();

            EXPECT_TRUE(bit_equal(out_late[0], out_fixed[0]))
                << "late " << out_late[0] << ", fixed " << out_fixed[0];
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_GraphPrimitivesReduceLateCount,
                                 ::testing::Values(std::size_t{1}, K_TILE - 1, K_TILE, K_TILE + 1,
                                                   K_LATE_CAPACITY));

        /// @brief Runs a late-bound Sum of ones whose live count changes to one value.
        class Medium_GraphPrimitivesReduceChangingCount : public ReduceAtCount
        {
        };

        /** @brief Tests that a live count changed between runs gives the sum at the new count. */
        TEST_P(Medium_GraphPrimitivesReduceChangingCount, LateBoundChangingCountGivesTheLiveSum)
        {
            SushiRuntime::Runtime& runtime = SushiBLAS::Test::TestEnvironment::runtime();
            auto in = buffer_holding(runtime, std::vector<double>(K_CHANGING_CAPACITY, 1.0));
            auto out = runtime.buffer<double>(1).value();

            std::size_t live_n = K_FIRST_LIVE_COUNT;
            auto graph = runtime.graph();
            reduce(graph.recorder(), in, out,
                   LateCount::from_host(K_CHANGING_CAPACITY, [&live_n]() { return live_n; }),
                   Sum<double>{}).value();

            // Sums of ones below 2^53 are exact, so the comparisons below are exact.
            graph.run().value();
            ASSERT_EQ(out[0], static_cast<double>(K_FIRST_LIVE_COUNT));

            live_n = GetParam();
            graph.run().value();
            EXPECT_EQ(out[0], static_cast<double>(GetParam()));
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_GraphPrimitivesReduceChangingCount,
                                 ::testing::Values(K_CHANGING_CAPACITY, std::size_t{3},
                                                   std::size_t{0}));

        /** @brief Tests that a live count that changes between runs keeps one compile. */
        TEST(Medium_GraphPrimitivesReduce, LateBoundChangingCountKeepsOneCompile)
        {
            SushiRuntime::Runtime& runtime = SushiBLAS::Test::TestEnvironment::runtime();
            auto in = buffer_holding(runtime, std::vector<double>(K_CHANGING_CAPACITY, 1.0));
            auto out = runtime.buffer<double>(1).value();

            std::size_t live_n = K_FIRST_LIVE_COUNT;
            auto graph = runtime.graph();
            reduce(graph.recorder(), in, out,
                   LateCount::from_host(K_CHANGING_CAPACITY, [&live_n]() { return live_n; }),
                   Sum<double>{}).value();

            graph.run().value();
            live_n = K_CHANGING_CAPACITY;
            graph.run().value();

            EXPECT_EQ(graph.compile_count(), 1u)
                << "a live count change between runs must not trigger a recompile";
        }

        /** @brief Tests that a live count above its capacity fails the run. */
        TEST(Medium_GraphPrimitivesReduce, LateBoundAboveCapacityFailsTheRun)
        {
            SushiRuntime::Runtime& runtime = SushiBLAS::Test::TestEnvironment::runtime();
            auto in = runtime.buffer<double>(100).value();
            auto out = runtime.buffer<double>(1).value();

            auto graph = runtime.graph();
            reduce(graph.recorder(), in, out,
                   LateCount::from_host(100, []() { return std::size_t{101}; }), Sum<double>{})
                .value();
            EXPECT_FALSE(graph.run().has_value());
        }

        /** @brief Tests that a late-bound reduce folds the count its predecessor wrote that run. */
        TEST(Medium_GraphPrimitivesReduce, DeviceCountIsReadAfterItsProducer)
        {
            constexpr std::size_t K_CAPACITY = 1000;
            using SushiRuntime::DeviceIndex;
            using SushiRuntime::Residency;

            SushiRuntime::Runtime& runtime = SushiBLAS::Test::TestEnvironment::runtime();
            auto values = buffer_holding(runtime, std::vector<double>(K_CAPACITY, 1.0));
            auto counter =
                runtime.buffer<std::uint32_t>(1, DeviceIndex{0}, Residency::Host).value();
            auto source = runtime.buffer<std::uint32_t>(1, DeviceIndex{0}, Residency::Host).value();
            auto out = runtime.buffer<double>(1).value();

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
        }

        /** @brief Tests that each level node launches at the out_count of its level. */
        TEST(Small_GraphPrimitivesReduce, LevelWidthsMatchTheLevelPlan)
        {
            const std::vector<std::size_t> expected_widths = {3u, 1u};

            auto values = HostRecorder::buffer<float>(K_TWO_LEVEL_COUNT);
            auto total = HostRecorder::buffer<float>(1);

            GraphPrimitives::HostPrimitiveRecorder recorder;
            GraphPrimitives::reduce(recorder, values, total, K_TWO_LEVEL_COUNT, Sum<float>{})
                .value();

            EXPECT_EQ(HostNodes::widths_of(recorder), expected_widths);
        }

        /** @brief Tests that the levels of one call record as one rule. */
        TEST(Small_GraphPrimitivesReduce, TheLevelsOfOneCallAreOneRule)
        {
            auto values = HostRecorder::buffer<float>(K_TWO_LEVEL_COUNT);
            auto total = HostRecorder::buffer<float>(1);

            GraphPrimitives::HostPrimitiveRecorder recorder;
            GraphPrimitives::reduce(recorder, values, total, K_TWO_LEVEL_COUNT, Sum<float>{})
                .value();

            EXPECT_EQ(recorder.rule_count(), 1u);
        }

        /** @brief Tests that a two-level chain totals every value of its input. */
        TEST(Medium_GraphPrimitivesReduce, TwoLevelChainTotalsEveryValue)
        {
            SushiRuntime::Runtime& runtime = SushiBLAS::Test::TestEnvironment::runtime();
            auto values = buffer_holding(runtime, std::vector<float>(K_TWO_LEVEL_COUNT, 1.0f));
            auto total = runtime.buffer<float>(1).value();

            auto graph = runtime.graph();
            reduce(graph.recorder(), values, total, K_TWO_LEVEL_COUNT, Sum<float>{}).value();
            graph.run().value();

            EXPECT_EQ(total[0], static_cast<float>(K_TWO_LEVEL_COUNT));
        }

        /** @brief Tests that a fixed count of zero records one node of width one. */
        TEST(Small_GraphPrimitivesReduce, ZeroCountStampsWidthOne)
        {
            auto values = HostRecorder::buffer<float>(1);
            auto total = HostRecorder::buffer<float>(1);

            GraphPrimitives::HostPrimitiveRecorder recorder;
            GraphPrimitives::reduce(recorder, values, total, std::size_t{0}, Sum<float>{})
                .value();

            EXPECT_EQ(HostNodes::widths_of(recorder), std::vector<std::size_t>{1u});
        }

        /// @brief Gives a reduce test one count form to record with.
        class ReduceAtForm : public ::testing::TestWithParam<CountForm>
        {
        };

        /// @brief Runs a reduce over windows and one over compact buffers in one count form.
        class Medium_GraphPrimitivesReduceWindow : public ReduceAtForm
        {
        };

        /** @brief Tests that a reduce over windows equals one over compact buffers. */
        TEST_P(Medium_GraphPrimitivesReduceWindow, WindowMatchesACompactBufferExactly)
        {
            const std::vector<float> host = order_sensitive_input(K_WINDOW_COUNT);

            SushiRuntime::Runtime& runtime = SushiBLAS::Test::TestEnvironment::runtime();
            auto wide_in = buffer_holding(
                runtime, framed(host, K_WINDOW_INPUT_OFFSET, K_WINDOW_INPUT_TAIL, K_INPUT_CANARY));
            auto wide_out = runtime.buffer<float>(K_WINDOW_OUTPUT_SIZE).value();
            auto compact_in = buffer_holding(runtime, host);
            auto compact_out = runtime.buffer<float>(1).value();

            auto graph = runtime.graph();
            record_sum(graph, wide_in, wide_out, GetParam(), K_WINDOW_COUNT,
                       ReduceWindow{K_WINDOW_INPUT_OFFSET, K_WINDOW_OUTPUT_OFFSET}).value();
            record_sum(graph, compact_in, compact_out, GetParam(), K_WINDOW_COUNT).value();
            graph.run().value();

            EXPECT_TRUE(bit_equal(wide_out[K_WINDOW_OUTPUT_OFFSET], compact_out[0]))
                << "got " << wide_out[K_WINDOW_OUTPUT_OFFSET] << ", expected " << compact_out[0];
        }

        /** @brief Tests that a reduce over a window writes no output element outside it. */
        TEST_P(Medium_GraphPrimitivesReduceWindow, WindowLeavesTheOutputAroundItUntouched)
        {
            SushiRuntime::Runtime& runtime = SushiBLAS::Test::TestEnvironment::runtime();
            auto wide_in = buffer_holding(
                runtime, framed(order_sensitive_input(K_WINDOW_COUNT), K_WINDOW_INPUT_OFFSET,
                                K_WINDOW_INPUT_TAIL, K_INPUT_CANARY));
            auto wide_out = buffer_holding(
                runtime, std::vector<float>(K_WINDOW_OUTPUT_SIZE, K_OUTPUT_CANARY));

            auto graph = runtime.graph();
            record_sum(graph, wide_in, wide_out, GetParam(), K_WINDOW_COUNT,
                       ReduceWindow{K_WINDOW_INPUT_OFFSET, K_WINDOW_OUTPUT_OFFSET}).value();
            graph.run().value();

            EXPECT_EQ(elements_except(wide_out, K_WINDOW_OUTPUT_OFFSET),
                      std::vector<float>(K_WINDOW_OUTPUT_SIZE - 1, K_OUTPUT_CANARY));
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_GraphPrimitivesReduceWindow,
                                 ::testing::Values(CountForm::Fixed, CountForm::Late),
                                 form_param_name);

        /// @brief Runs one probe of a part of an operand in one count form.
        class Small_GraphPrimitivesReduceWindowOrder
            : public ::testing::TestWithParam<std::tuple<ProbeRow, CountForm>>
        {
        };

        /** @brief Tests that a writer of an operand waits only when it hits the window. */
        TEST_P(Small_GraphPrimitivesReduceWindowOrder, WindowOrdersOnlyOverlappingWriters)
        {
            const ProbeRow& row = std::get<0>(GetParam());

            auto in = HostRecorder::buffer<float>(K_PROBED_OFFSET + K_PROBED_COUNT + K_PROBED_TAIL);
            auto out = HostRecorder::buffer<float>(4);

            GraphPrimitives::HostPrimitiveRecorder recorder;
            record_sum(recorder, in, out, std::get<1>(GetParam()), K_PROBED_COUNT,
                       ReduceWindow{K_PROBED_OFFSET, 1}).value();
            const std::size_t waits = recorder.nodes_touching(probed_region(row, in, out));

            EXPECT_EQ(waits > 0u, row.ordered) << "the probe waits for " << waits << " nodes";
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesReduceWindowOrder,
                                 ::testing::Combine(::testing::ValuesIn(probe_rows()),
                                                    ::testing::Values(CountForm::Fixed,
                                                                      CountForm::Late)),
                                 row_form_name<ProbeRow>);

        /// @brief Runs one shape of a reduce named "sum" with the Bitwise class.
        class Small_GraphPrimitivesReduceNamedCall : public ::testing::TestWithParam<NamedShapeRow>
        {
        };

        /** @brief Tests that a named reduce names every node by its stage. */
        TEST_P(Small_GraphPrimitivesReduceNamedCall, NamedCallNamesItsNodesByStage)
        {
            const NamedShapeRow& row = GetParam();

            auto values = HostRecorder::buffer<float>(K_TWO_LEVEL_COUNT);
            auto total = HostRecorder::buffer<float>(1);

            GraphPrimitives::HostPrimitiveRecorder recorder;
            record_sum(recorder, values, total, row.form, row.count, ReduceWindow{},
                       sum_options())
                .value();

            EXPECT_EQ(HostNodes::names_of(recorder), row.names);
        }

        /** @brief Tests that a named reduce states its class on every node. */
        TEST_P(Small_GraphPrimitivesReduceNamedCall, NamedCallStatesTheClassOfEveryNode)
        {
            const NamedShapeRow& row = GetParam();

            auto values = HostRecorder::buffer<float>(K_TWO_LEVEL_COUNT);
            auto total = HostRecorder::buffer<float>(1);

            GraphPrimitives::HostPrimitiveRecorder recorder;
            record_sum(recorder, values, total, row.form, row.count, ReduceWindow{},
                       sum_options())
                .value();
            const std::vector<Determinism> classes = HostNodes::determinisms_of(recorder);

            ASSERT_FALSE(classes.empty());
            EXPECT_EQ(classes, std::vector<Determinism>(classes.size(), Determinism::Bitwise));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesReduceNamedCall,
                                 ::testing::ValuesIn(named_shape_rows()), row_name<NamedShapeRow>);

        /// @brief Runs a reduce with no options in one count form.
        class Small_GraphPrimitivesReduceUnnamedCall : public ReduceAtForm
        {
        };

        /** @brief Tests that a call with no options keeps its nodes unnamed. */
        TEST_P(Small_GraphPrimitivesReduceUnnamedCall, UnnamedCallKeepsItsNodesUnnamed)
        {
            auto values = HostRecorder::buffer<float>(K_TWO_LEVEL_COUNT);
            auto total = HostRecorder::buffer<float>(1);

            GraphPrimitives::HostPrimitiveRecorder recorder;
            record_sum(recorder, values, total, GetParam(), K_TWO_LEVEL_COUNT).value();
            const std::vector<std::string> names = HostNodes::names_of(recorder);

            ASSERT_FALSE(names.empty());
            EXPECT_EQ(names, std::vector<std::string>(names.size(), std::string()));
        }

        /** @brief Tests that a call with no options leaves the class of every node Unstated. */
        TEST_P(Small_GraphPrimitivesReduceUnnamedCall, UnnamedCallLeavesItsClassUnstated)
        {
            auto values = HostRecorder::buffer<float>(K_TWO_LEVEL_COUNT);
            auto total = HostRecorder::buffer<float>(1);

            GraphPrimitives::HostPrimitiveRecorder recorder;
            record_sum(recorder, values, total, GetParam(), K_TWO_LEVEL_COUNT).value();
            const std::vector<Determinism> classes = HostNodes::determinisms_of(recorder);

            ASSERT_FALSE(classes.empty());
            EXPECT_EQ(classes, std::vector<Determinism>(classes.size(), Determinism::Unstated));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesReduceUnnamedCall,
                                 ::testing::Values(CountForm::Fixed, CountForm::Late),
                                 form_param_name);

    } // namespace Test
} // namespace SushiBLAS
