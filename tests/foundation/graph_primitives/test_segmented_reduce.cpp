/**************************************************************************/
/* test_segmented_reduce.cpp                                              */
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
 * @file test_segmented_reduce.cpp
 * @brief Tests segmented_reduce(): its per-segment fold, its late-bound count and its widths.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <array>
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
#include <SushiRuntime/api/vocabulary/late_count.hpp>
#include <SushiRuntime/core/error.hpp>

#include <SushiBLAS/graph_primitives/combiners.hpp>
#include <SushiBLAS/graph_primitives/host_primitive_recorder.hpp>
#include <SushiBLAS/graph_primitives/primitive_options.hpp>
#include <SushiBLAS/graph_primitives/segmented_reduce.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include "device_sized_harness.hpp"
#include "host_nodes.hpp"
#include "window_probe.hpp"
#include "runtime_port_calls.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using GraphPrimitives::PrimitiveOptions;
            using GraphPrimitives::SegmentedReduceWindow;
            using GraphPrimitives::Sum;
            using RuntimePort::segmented_reduce;
            using SushiRuntime::Buffer;
            using SushiRuntime::LateCount;
            using SushiRuntime::Runtime;

            /// @brief Holds a value no segment of these tests folds to.
            constexpr float K_UNWRITTEN = -1.0f;

            /// @brief Holds the number of segments of the ragged case.
            constexpr std::size_t K_RAGGED_SEGMENTS = 5;

            /** @brief Holds ragged segment bounds with an empty segment and a long one. */
            constexpr std::array<std::uint32_t, K_RAGGED_SEGMENTS + 1> K_RAGGED_OFFSETS = {
                0, 3, 3, 10, 11, 1011};

            /// @brief Holds the segment of the ragged case that indexes no value.
            constexpr std::size_t K_EMPTY_SEGMENT = 1;

            /// @brief Holds the values the ragged case indexes.
            constexpr std::size_t K_RAGGED_VALUES = K_RAGGED_OFFSETS.back();

            /** @brief Names the two forms of the segment count a call can take. */
            enum class CountForm
            {
                Fixed,    /**< A std::size_t known when the call records. */
                LateBound /**< A LateCount resolved when the graph runs. */
            };

            /** @brief Returns the name of a count form in a test report. */
            const char* form_name(CountForm form)
            {
                return (form == CountForm::LateBound) ? "LateBound" : "Fixed";
            }

            /** @brief Returns both count forms as test parameters. */
            auto both_forms()
            {
                return ::testing::Values(CountForm::Fixed, CountForm::LateBound);
            }

            /** @brief Names a row whose parameter is a count form alone. */
            std::string form_param_name(const ::testing::TestParamInfo<CountForm>& info)
            {
                return form_name(info.param);
            }

            /** @brief Names a row whose parameter is a count form and a named row. */
            template <typename Row>
            std::string formed_name(const ::testing::TestParamInfo<std::tuple<CountForm, Row>>& info)
            {
                return std::string(form_name(std::get<0>(info.param))) + "_"
                       + std::get<1>(info.param).name;
            }

            /** @brief Names a row by its own name field. */
            template <typename Row>
            std::string row_name(const ::testing::TestParamInfo<Row>& info)
            {
                return info.param.name;
            }

            /** @brief Returns the bit pattern of a float. */
            std::uint32_t bit_pattern(float value)
            {
                std::uint32_t pattern = 0;
                std::memcpy(&pattern, &value, sizeof(pattern));
                return pattern;
            }

            /** @brief Returns the bit pattern of each float, in order. */
            std::vector<std::uint32_t> bits_of(const std::vector<float>& values)
            {
                std::vector<std::uint32_t> patterns;
                patterns.reserve(values.size());
                for (const float value : values)
                    patterns.push_back(bit_pattern(value));
                return patterns;
            }

            /** @brief Returns elements [offset, offset + count) of a host-addressable buffer. */
            template <class B>
            std::vector<GraphPrimitives::Detail::BufferValue<B>>
            read_buffer(const B& buffer, std::size_t offset, std::size_t count)
            {
                std::vector<GraphPrimitives::Detail::BufferValue<B>> values;
                values.reserve(count);
                for (std::size_t i = 0; i < count; ++i)
                    values.push_back(buffer[offset + i]);
                return values;
            }

            /** @brief Sets every element of a host-addressable buffer to value. */
            template <class B>
            void fill_constant(B& buffer, GraphPrimitives::Detail::BufferValue<B> value)
            {
                for (std::size_t i = 0; i < buffer.size(); ++i)
                    buffer[i] = value;
            }

            /** @brief Returns the value at index i of the ragged case. */
            float ragged_value(std::size_t i)
            {
                return static_cast<float>(i % 13) * 0.25f;
            }

            /** @brief Returns the left-to-right sum of one segment, zero for an empty one. */
            float host_segment_sum(std::size_t begin, std::size_t end)
            {
                if (end <= begin)
                    return 0.0f;
                float sum = ragged_value(begin);
                for (std::size_t j = begin + 1; j < end; ++j)
                    sum = sum + ragged_value(j);
                return sum;
            }

            /** @brief Returns the host fold of each ragged segment below live, K_UNWRITTEN from live on. */
            std::vector<float> ragged_sums(std::size_t live = K_RAGGED_SEGMENTS)
            {
                std::vector<float> sums(K_RAGGED_SEGMENTS, K_UNWRITTEN);
                for (std::size_t s = 0; s < live; ++s)
                    sums[s] = host_segment_sum(K_RAGGED_OFFSETS[s], K_RAGGED_OFFSETS[s + 1]);
                return sums;
            }

            /** @brief Holds the ragged case in compact buffers, its output unwritten. */
            struct RaggedSegments
            {
                Buffer<float> values;         /**< The values the segments index. */
                Buffer<std::uint32_t> bounds; /**< The segment offsets. */
                Buffer<float> out;            /**< One result per segment. */

                /** @brief Allocates the buffers and writes the values, bounds and K_UNWRITTEN. */
                explicit RaggedSegments(Runtime& runtime)
                    : values(runtime.buffer<float>(K_RAGGED_VALUES).value()),
                      bounds(runtime.buffer<std::uint32_t>(K_RAGGED_OFFSETS.size()).value()),
                      out(runtime.buffer<float>(K_RAGGED_SEGMENTS).value())
                {
                    for (std::size_t i = 0; i < K_RAGGED_VALUES; ++i)
                        values[i] = ragged_value(i);
                    for (std::size_t i = 0; i < K_RAGGED_OFFSETS.size(); ++i)
                        bounds[i] = K_RAGGED_OFFSETS[i];
                    fill_constant(out, K_UNWRITTEN);
                }
            };

            /// @brief Names the recorder whose operands the host rows use.
            using HostRecorder = GraphPrimitives::HostPrimitiveRecorder;

            /// @brief Names the uint32 operand of a host recorder.
            using HostWords = GraphPrimitives::HostBuffer<std::uint32_t>;

            /** @brief Holds one-value segments in offsets and output padded past the count. */
            struct UnitSegments
            {
                static constexpr std::size_t K_OFFSETS_PAD = 9; /**< Offsets past the count. */
                static constexpr std::size_t K_OUTPUT_PAD = 8;  /**< Outputs past the count. */

                HostWords offsets; /**< Bounds s at index s, then padding. */
                HostWords values;  /**< One per segment, every one 1. */
                HostWords out;     /**< One result per segment, then padding. */

                /** @brief Allocates segments one-value segments and writes their bounds and values. */
                explicit UnitSegments(std::size_t segments)
                    : offsets(HostRecorder::buffer<std::uint32_t>(segments + K_OFFSETS_PAD)),
                      values(HostRecorder::buffer<std::uint32_t>(segments)),
                      out(HostRecorder::buffer<std::uint32_t>(segments + K_OUTPUT_PAD))
                {
                    for (std::size_t s = 0; s <= segments; ++s)
                        offsets[s] = static_cast<std::uint32_t>(s);
                    fill_constant(values, 1u);
                }
            };

            /** @brief Writes bounds whose segment lengths cycle through 0 to 4; returns the end. */
            std::uint32_t fill_cycling_offsets(Buffer<std::uint32_t>& offsets)
            {
                std::uint32_t at = 0;
                for (std::size_t s = 0; s < offsets.size(); ++s)
                {
                    offsets[s] = at;
                    at += static_cast<std::uint32_t>(s % 5);
                }
                return offsets[offsets.size() - 1];
            }

            /** @brief Writes a multiplicative hash of each index into values. */
            void fill_hashed(Buffer<std::uint32_t>& values)
            {
                for (std::size_t i = 0; i < values.size(); ++i)
                    values[i] = static_cast<std::uint32_t>(i) * 2654435761u;
            }

            /** @brief Records a segmented sum in the given count form. */
            template <typename T>
            SushiRuntime::Result<void>
            record_reduce(SushiRuntime::Recorder& recorder, const Buffer<T>& values,
                          const Buffer<std::uint32_t>& bounds, const Buffer<T>& out,
                          CountForm form, std::size_t segments,
                          const SegmentedReduceWindow& window = SegmentedReduceWindow{},
                          const std::optional<PrimitiveOptions>& options = std::nullopt)
            {
                if (form == CountForm::LateBound)
                    return segmented_reduce(recorder, values, bounds, out,
                                            LateCount::fixed(segments), Sum<T>{}, window,
                                            options);

                return segmented_reduce(recorder, values, bounds, out, segments, Sum<T>{},
                                        window, options);
            }

            /** @brief Records a segmented sum on a host recorder in the given count form. */
            template <typename T>
            SushiRuntime::Result<void>
            record_reduce(GraphPrimitives::HostPrimitiveRecorder& recorder,
                          const GraphPrimitives::HostBuffer<T>& values, const HostWords& bounds,
                          const GraphPrimitives::HostBuffer<T>& out, CountForm form,
                          std::size_t segments,
                          const SegmentedReduceWindow& window = SegmentedReduceWindow{},
                          const std::optional<PrimitiveOptions>& options = std::nullopt)
            {
                if (form == CountForm::LateBound)
                    return GraphPrimitives::segmented_reduce(recorder, values, bounds, out,
                                                             LateCount::fixed(segments),
                                                             Sum<T>{}, window, options);

                return GraphPrimitives::segmented_reduce(recorder, values, bounds, out, segments,
                                                         Sum<T>{}, window, options);
            }

            /** @brief Holds the ragged case inside wide buffers and alone. */
            struct WindowedSegments
            {
                static constexpr std::size_t K_VALUES_OFFSET = 5; /**< Values window start. */
                static constexpr std::size_t K_BOUNDS_OFFSET = 3; /**< Offsets window start. */
                static constexpr std::size_t K_OUTPUT_OFFSET = 4; /**< Output window start. */
                static constexpr std::size_t K_TAIL = 7;          /**< Elements past a window. */

                Buffer<float> wide_values;          /**< Values inside a window. */
                Buffer<std::uint32_t> wide_bounds;  /**< Offsets inside a window. */
                Buffer<float> wide_out;             /**< Output inside a window. */
                Buffer<float> values;               /**< Values alone. */
                Buffer<std::uint32_t> bounds;       /**< Offsets alone. */
                Buffer<float> out;                  /**< Output alone. */

                /** @brief Allocates both layouts and writes the same values into each. */
                explicit WindowedSegments(Runtime& runtime)
                    : wide_values(runtime.buffer<float>(wide_value_count()).value()),
                      wide_bounds(runtime.buffer<std::uint32_t>(
                                      K_BOUNDS_OFFSET + K_RAGGED_OFFSETS.size() + K_TAIL).value()),
                      wide_out(runtime.buffer<float>(K_OUTPUT_OFFSET + segments() + K_TAIL)
                                   .value()),
                      values(runtime.buffer<float>(value_count()).value()),
                      bounds(runtime.buffer<std::uint32_t>(K_RAGGED_OFFSETS.size()).value()),
                      out(runtime.buffer<float>(segments()).value())
                {
                    fill_constant(wide_values, 1.0e6f);
                    fill_constant(wide_bounds, 0u);
                    fill_constant(wide_out, K_UNWRITTEN);
                    for (std::size_t i = 0; i < value_count(); ++i)
                    {
                        values[i] = static_cast<float>(i % 13) * 0.25f + 1.0e-3f;
                        wide_values[K_VALUES_OFFSET + i] = values[i];
                    }
                    for (std::size_t i = 0; i < K_RAGGED_OFFSETS.size(); ++i)
                    {
                        bounds[i] = K_RAGGED_OFFSETS[i];
                        wide_bounds[K_BOUNDS_OFFSET + i] = K_RAGGED_OFFSETS[i];
                    }
                }

                /** @brief Returns the segments of the ragged case. */
                [[nodiscard]] static constexpr std::size_t segments() { return K_RAGGED_SEGMENTS; }

                /** @brief Returns the values the ragged case indexes. */
                [[nodiscard]] static constexpr std::size_t value_count() { return K_RAGGED_VALUES; }

                /** @brief Returns the size of the wide values buffer. */
                [[nodiscard]] static constexpr std::size_t wide_value_count()
                {
                    return K_VALUES_OFFSET + value_count() + K_TAIL;
                }

                /** @brief Returns the window that names the ragged case inside the wide buffers. */
                [[nodiscard]] static SegmentedReduceWindow window()
                {
                    return SegmentedReduceWindow{K_VALUES_OFFSET, value_count(), K_BOUNDS_OFFSET,
                                                 K_OUTPUT_OFFSET};
                }
            };

            /** @brief Holds the wide operands of a windowed call on a host recorder. */
            struct HostWindowedSegments
            {
                GraphPrimitives::HostBuffer<float> wide_values; /**< Values inside a window. */
                HostWords wide_bounds;                          /**< Offsets inside a window. */
                GraphPrimitives::HostBuffer<float> wide_out;    /**< Output inside a window. */

                /** @brief Allocates the wide operands at the size WindowedSegments gives them. */
                HostWindowedSegments()
                    : wide_values(HostRecorder::buffer<float>(
                          WindowedSegments::wide_value_count())),
                      wide_bounds(HostRecorder::buffer<std::uint32_t>(
                          WindowedSegments::K_BOUNDS_OFFSET + K_RAGGED_OFFSETS.size() +
                          WindowedSegments::K_TAIL)),
                      wide_out(HostRecorder::buffer<float>(
                          WindowedSegments::K_OUTPUT_OFFSET + WindowedSegments::segments() +
                          WindowedSegments::K_TAIL))
                {
                }
            };

            /** @brief Records a segmented reduction of the wide buffers on a host recorder. */
            SushiRuntime::Result<void>
            record_windowed(GraphPrimitives::HostPrimitiveRecorder& recorder,
                            const HostWindowedSegments& buffers, CountForm form,
                            const SegmentedReduceWindow& window)
            {
                return record_reduce(recorder, buffers.wide_values, buffers.wide_bounds,
                                     buffers.wide_out, form, WindowedSegments::segments(), window);
            }

            /** @brief Records a segmented reduction of the wide buffers through window. */
            SushiRuntime::Result<void>
            record_windowed(SushiRuntime::Recorder& recorder, const WindowedSegments& buffers,
                            CountForm form, const SegmentedReduceWindow& window)
            {
                return record_reduce(recorder, buffers.wide_values, buffers.wide_bounds,
                                     buffers.wide_out, form, WindowedSegments::segments(), window);
            }

            /** @brief Records the windowed and the compact reduction in one graph and runs it. */
            void run_windowed_and_compact(Runtime& runtime, const WindowedSegments& buffers,
                                          CountForm form)
            {
                auto graph = runtime.graph();
                record_windowed(graph.recorder(), buffers, form, WindowedSegments::window())
                    .value();
                segmented_reduce(graph.recorder(), buffers.values, buffers.bounds, buffers.out,
                                 WindowedSegments::segments(), Sum<float>{}).value();
                graph.run().value();
            }

            /** @brief Names the operand a window probe writes into. */
            enum class WindowOperand
            {
                Values,  /**< The wide values buffer. */
                Offsets, /**< The wide offsets buffer. */
                Output   /**< The wide output buffer. */
            };

            /// @brief Holds the end of the values window inside the wide values buffer.
            constexpr std::size_t K_VALUES_END =
                WindowedSegments::K_VALUES_OFFSET + WindowedSegments::value_count();

            /// @brief Holds the end of the output window inside the wide output buffer.
            constexpr std::size_t K_OUTPUT_END =
                WindowedSegments::K_OUTPUT_OFFSET + WindowedSegments::segments();

            /** @brief Holds a region a later node writes and whether it must wait for the call. */
            struct ProbeRow
            {
                const char* name;      /**< Names the row in the test report. */
                WindowOperand operand; /**< The buffer the region lies in. */
                std::size_t offset;    /**< First element of the region. */
                std::size_t count;     /**< Elements of the region. */
                bool ordered;          /**< Whether the writer waits for the call. */
            };

            /** @brief Returns the region a probe row names inside the wide buffers. */
            SushiRuntime::BufferRegion probed_region(const HostWindowedSegments& buffers,
                                                     const ProbeRow& row)
            {
                if (row.operand == WindowOperand::Values)
                    return WindowProbe::region_of(buffers.wide_values, row.offset, row.count);
                if (row.operand == WindowOperand::Offsets)
                    return WindowProbe::region_of(buffers.wide_bounds, row.offset, row.count);

                return WindowProbe::region_of(buffers.wide_out, row.offset, row.count);
            }

            /** @brief Returns the regions probed around and inside each window. */
            std::vector<ProbeRow> probe_rows()
            {
                return {
                    {"ValuesBeforeTheWindow", WindowOperand::Values, 0,
                     WindowedSegments::K_VALUES_OFFSET, false},
                    {"ValuesAfterTheWindow", WindowOperand::Values, K_VALUES_END, 1, false},
                    {"FirstValueOfTheWindow", WindowOperand::Values,
                     WindowedSegments::K_VALUES_OFFSET, 1, true},
                    {"OffsetsBeforeTheWindow", WindowOperand::Offsets, 0,
                     WindowedSegments::K_BOUNDS_OFFSET, false},
                    {"FirstOffsetOfTheWindow", WindowOperand::Offsets,
                     WindowedSegments::K_BOUNDS_OFFSET, 1, true},
                    {"OutputAfterTheWindow", WindowOperand::Output, K_OUTPUT_END, 1, false},
                    {"LastOutputOfTheWindow", WindowOperand::Output, K_OUTPUT_END - 1, 1, true},
                };
            }

            /** @brief Holds the small operands of a call that is recorded but never run. */
            struct SmallCall
            {
                /// @brief Holds the values.
                GraphPrimitives::HostBuffer<float> values = HostRecorder::buffer<float>(8);
                /// @brief Holds the segment offsets.
                HostWords bounds = HostRecorder::buffer<std::uint32_t>(3);
                /// @brief Holds the output.
                GraphPrimitives::HostBuffer<float> out = HostRecorder::buffer<float>(2);

                /** @brief Records a two-segment sum on a host recorder in the given form. */
                SushiRuntime::Result<void>
                record_host(GraphPrimitives::HostPrimitiveRecorder& recorder, CountForm form,
                            const std::optional<PrimitiveOptions>& options = std::nullopt) const
                {
                    return record_reduce(recorder, values, bounds, out, form, std::size_t{2},
                                         SegmentedReduceWindow{}, options);
                }
            };

            /// @brief Holds the options every named call states.
            const PrimitiveOptions K_NAMED{"gather", SushiRuntime::Determinism::Bitwise};

            /** @brief Holds a count form and the names its named call gives its nodes. */
            struct LabelRow
            {
                const char* name;                     /**< Names the row in the test report. */
                CountForm form;                       /**< The form of the call. */
                std::vector<std::string> stage_names; /**< One name per node, in order. */
            };

            /** @brief Returns the node names of a call named "gather", per count form. */
            std::vector<LabelRow> label_rows()
            {
                return {{"Fixed", CountForm::Fixed, {"gather"}},
                        {"LateBound", CountForm::LateBound, {"gather.resolve", "gather.segment"}}};
            }

        } // namespace

        /** @brief Tests a segmented reduce over ragged segments, an empty one included. */
        TEST(Medium_GraphPrimitivesSegmentedReduce, FoldsRaggedSegmentsLeftToRight)
        {
            SushiRuntime::Runtime& runtime = SushiBLAS::Test::TestEnvironment::runtime();
            RaggedSegments ragged(runtime);

            auto graph = runtime.graph();
            segmented_reduce(graph.recorder(), ragged.values, ragged.bounds, ragged.out,
                             K_RAGGED_SEGMENTS, Sum<float>{}).value();
            graph.run().value();

            EXPECT_EQ(bits_of(read_buffer(ragged.out, 0, K_RAGGED_SEGMENTS)),
                      bits_of(ragged_sums()));
            EXPECT_EQ(bit_pattern(ragged.out[K_EMPTY_SEGMENT]), bit_pattern(0.0f))
                << "an empty segment folds to zero";
        }

        /// @brief Runs a late-bound call at one live segment count.
        class Medium_GraphPrimitivesSegmentedReduceLiveCount
            : public ::testing::TestWithParam<std::size_t>
        {
        };

        /** @brief Tests that a live segment count below capacity leaves the rest unwritten. */
        TEST_P(Medium_GraphPrimitivesSegmentedReduceLiveCount,
               LateBoundCountFoldsOnlyTheLiveSegments)
        {
            const std::size_t live = GetParam();
            SushiRuntime::Runtime& runtime = SushiBLAS::Test::TestEnvironment::runtime();
            RaggedSegments ragged(runtime);

            std::size_t live_segments = K_RAGGED_SEGMENTS;
            const LateCount count = LateCount::from_host(
                K_RAGGED_SEGMENTS, [&live_segments]() { return live_segments; });
            auto graph = runtime.graph();
            segmented_reduce(graph.recorder(), ragged.values, ragged.bounds, ragged.out, count,
                             Sum<float>{}).value();

            // A first run at capacity, so the second run shows whether a new count recompiles.
            graph.run().value();
            live_segments = live;
            fill_constant(ragged.out, K_UNWRITTEN);
            graph.run().value();

            EXPECT_EQ(bits_of(read_buffer(ragged.out, 0, K_RAGGED_SEGMENTS)),
                      bits_of(ragged_sums(live)));
            EXPECT_EQ(graph.compile_count(), 1u);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_GraphPrimitivesSegmentedReduceLiveCount,
                                 ::testing::Values(std::size_t{0}, std::size_t{2}, std::size_t{4},
                                                   K_RAGGED_SEGMENTS),
                                 [](const ::testing::TestParamInfo<std::size_t>& info)
                                 { return "Live" + std::to_string(info.param); });

        /** @brief Tests that the fixed-count node launches at segment_count, not a buffer size. */
        TEST(Small_GraphPrimitivesSegmentedReduce, FixedCountWidthIsTheSegmentCount)
        {
            constexpr std::size_t K_SEGMENTS = 64;
            UnitSegments segments(K_SEGMENTS);

            GraphPrimitives::HostPrimitiveRecorder recorder;
            GraphPrimitives::segmented_reduce(recorder, segments.values, segments.offsets,
                                              segments.out, K_SEGMENTS, Sum<std::uint32_t>{})
                .value();
            recorder.run().value();

            const std::vector<std::size_t> expected_widths = {K_SEGMENTS};
            EXPECT_EQ(HostNodes::widths_of(recorder), expected_widths);
            EXPECT_EQ(read_buffer(segments.out, 0, K_SEGMENTS),
                      std::vector<std::uint32_t>(K_SEGMENTS, 1u));
        }

        /** @brief Tests that the late-bound form records a resolver and a node of capacity. */
        TEST(Small_GraphPrimitivesSegmentedReduce, LateBoundWidthsAreOneAndTheCapacity)
        {
            constexpr std::size_t K_SEGMENTS = 128;
            UnitSegments segments(K_SEGMENTS);

            GraphPrimitives::HostPrimitiveRecorder recorder;
            GraphPrimitives::segmented_reduce(recorder, segments.values, segments.offsets,
                                              segments.out, LateCount::fixed(K_SEGMENTS),
                                              Sum<std::uint32_t>{})
                .value();
            EXPECT_EQ(recorder.rule_count(), 1u) << "both nodes of one call are one rule";
            recorder.run().value();

            const std::vector<std::size_t> expected_widths = {1u, K_SEGMENTS};
            EXPECT_EQ(HostNodes::widths_of(recorder), expected_widths);
            EXPECT_EQ(read_buffer(segments.out, 0, K_SEGMENTS),
                      std::vector<std::uint32_t>(K_SEGMENTS, 1u));
        }

        /** @brief Tests segmented_reduce() with a device count on both layers. */
        TEST(Medium_GraphPrimitivesDeviceSized, SegmentedReduceChainMatchesScheduler)
        {
            using namespace DeviceSized;

            constexpr std::size_t K_SEGMENTS = 500;
            expect_chain_matches_scheduler([](Runtime& runtime)
            {
                Harness harness(runtime, K_SEGMENTS);
                auto offsets = runtime.buffer<std::uint32_t>(K_SEGMENTS + 1).value();
                auto values = runtime.buffer<std::uint32_t>(fill_cycling_offsets(offsets)).value();
                fill_hashed(values);
                auto out = runtime.buffer<std::uint32_t>(K_SEGMENTS).value();
                fill_constant(out, 0u);

                auto graph = runtime.graph();
                harness.add_producer(graph);
                RuntimePort::segmented_reduce(graph.recorder(), values, offsets, out,
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

        /// @brief Runs a windowed call beside a compact one in one count form.
        class Medium_GraphPrimitivesSegmentedReduceWindow
            : public ::testing::TestWithParam<CountForm>
        {
        };

        /** @brief Tests that a segmented reduce over windows equals one over compact buffers. */
        TEST_P(Medium_GraphPrimitivesSegmentedReduceWindow, WindowMatchesACompactBufferExactly)
        {
            constexpr std::size_t K_SEGMENTS = WindowedSegments::segments();
            SushiRuntime::Runtime& runtime = SushiBLAS::Test::TestEnvironment::runtime();
            const WindowedSegments buffers(runtime);

            run_windowed_and_compact(runtime, buffers, GetParam());

            EXPECT_EQ(bits_of(read_buffer(buffers.wide_out, WindowedSegments::K_OUTPUT_OFFSET,
                                          K_SEGMENTS)),
                      bits_of(read_buffer(buffers.out, 0, K_SEGMENTS)));
        }

        /** @brief Tests that a windowed call writes no output element outside its window. */
        TEST_P(Medium_GraphPrimitivesSegmentedReduceWindow, WindowLeavesTheOutputAroundItUnwritten)
        {
            constexpr std::size_t K_BEFORE = WindowedSegments::K_OUTPUT_OFFSET;
            constexpr std::size_t K_AFTER = WindowedSegments::K_TAIL;
            SushiRuntime::Runtime& runtime = SushiBLAS::Test::TestEnvironment::runtime();
            const WindowedSegments buffers(runtime);

            run_windowed_and_compact(runtime, buffers, GetParam());

            EXPECT_EQ(bits_of(read_buffer(buffers.wide_out, 0, K_BEFORE)),
                      bits_of(std::vector<float>(K_BEFORE, K_UNWRITTEN)))
                << "an element before the output window was written";
            EXPECT_EQ(bits_of(read_buffer(buffers.wide_out, K_OUTPUT_END, K_AFTER)),
                      bits_of(std::vector<float>(K_AFTER, K_UNWRITTEN)))
                << "an element after the output window was written";
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_GraphPrimitivesSegmentedReduceWindow, both_forms(),
                                 form_param_name);

        /// @brief Probes one region near a window in one count form.
        class Small_GraphPrimitivesSegmentedReduceWindowOrder
            : public ::testing::TestWithParam<std::tuple<CountForm, ProbeRow>>
        {
        };

        /** @brief Tests that a writer of an operand's buffer waits only when it hits the window. */
        TEST_P(Small_GraphPrimitivesSegmentedReduceWindowOrder, WindowOrdersOnlyOverlappingWriters)
        {
            const CountForm form = std::get<0>(GetParam());
            const ProbeRow& row = std::get<1>(GetParam());
            const HostWindowedSegments buffers;

            GraphPrimitives::HostPrimitiveRecorder recorder;
            record_windowed(recorder, buffers, form, WindowedSegments::window()).value();
            const std::size_t waits = recorder.nodes_touching(probed_region(buffers, row));

            EXPECT_EQ(waits > 0u, row.ordered) << "the writer waits for " << waits << " nodes";
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesSegmentedReduceWindowOrder,
                                 ::testing::Combine(both_forms(),
                                                    ::testing::ValuesIn(probe_rows())),
                                 formed_name<ProbeRow>);

        /// @brief Records one small call in one count form and reads its node labels.
        class Small_GraphPrimitivesSegmentedReduceLabels
            : public ::testing::TestWithParam<LabelRow>
        {
        };

        /** @brief Tests that a named call names each of its nodes by stage. */
        TEST_P(Small_GraphPrimitivesSegmentedReduceLabels, NamedCallNamesItsNodesByStage)
        {
            SmallCall call;
            GraphPrimitives::HostPrimitiveRecorder recorder;
            call.record_host(recorder, GetParam().form, K_NAMED).value();

            EXPECT_EQ(HostNodes::names_of(recorder), GetParam().stage_names);
        }

        /** @brief Tests that a named call states its class on every node it records. */
        TEST_P(Small_GraphPrimitivesSegmentedReduceLabels, NamedCallStatesTheClassOfEveryNode)
        {
            SmallCall call;
            GraphPrimitives::HostPrimitiveRecorder recorder;
            call.record_host(recorder, GetParam().form, K_NAMED).value();

            EXPECT_EQ(HostNodes::determinisms_of(recorder),
                      std::vector<SushiRuntime::Determinism>(GetParam().stage_names.size(),
                                                             SushiRuntime::Determinism::Bitwise));
        }

        /** @brief Tests that a call with no options keeps its nodes unnamed. */
        TEST_P(Small_GraphPrimitivesSegmentedReduceLabels, UnnamedCallKeepsItsNodesUnnamed)
        {
            SmallCall call;
            GraphPrimitives::HostPrimitiveRecorder recorder;
            call.record_host(recorder, GetParam().form).value();

            EXPECT_EQ(HostNodes::names_of(recorder),
                      std::vector<std::string>(GetParam().stage_names.size(), std::string()));
        }

        /** @brief Tests that a call with no options leaves the class of its nodes Unstated. */
        TEST_P(Small_GraphPrimitivesSegmentedReduceLabels, UnnamedCallLeavesItsNodesUnstated)
        {
            SmallCall call;
            GraphPrimitives::HostPrimitiveRecorder recorder;
            call.record_host(recorder, GetParam().form).value();

            EXPECT_EQ(HostNodes::determinisms_of(recorder),
                      std::vector<SushiRuntime::Determinism>(GetParam().stage_names.size(),
                                                             SushiRuntime::Determinism::Unstated));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesSegmentedReduceLabels,
                                 ::testing::ValuesIn(label_rows()), row_name<LabelRow>);

    } // namespace Test
} // namespace SushiBLAS
