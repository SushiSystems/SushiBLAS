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
 * @file test_segmented_reduce.cpp
 * @brief Tests segmented_reduce(): its per-segment fold, its late-bound count and its widths.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/graph/graph.hpp>
#include <SushiRuntime/api/runtime/runtime.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>
#include <SushiRuntime/core/error.hpp>

#include <SushiBLAS/graph_primitives/combiners.hpp>
#include <SushiBLAS/graph_primitives/segmented_reduce.hpp>
#include "device_sized_harness.hpp"
#include "window_probe.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using GraphPrimitives::Sum;
            using GraphPrimitives::segmented_reduce;
            using SushiRuntime::LateCount;
            using SushiRuntime::Runtime;

            /// @brief Holds a value no segment of these tests folds to.
            constexpr float K_UNWRITTEN = -1.0f;

            /** @brief Returns whether two floats have the same bit pattern. */
            bool bit_equal(float a, float b)
            {
                std::uint32_t pattern_a = 0;
                std::uint32_t pattern_b = 0;
                std::memcpy(&pattern_a, &a, sizeof(pattern_a));
                std::memcpy(&pattern_b, &b, sizeof(pattern_b));
                return pattern_a == pattern_b;
            }

            /** @brief Returns the left-to-right sum of one segment, zero for an empty one. */
            float host_segment_sum(const std::vector<float>& values, std::size_t begin,
                                   std::size_t end)
            {
                if (end <= begin)
                    return 0.0f;
                float sum = values[begin];
                for (std::size_t j = begin + 1; j < end; ++j)
                    sum = sum + values[j];
                return sum;
            }

            /** @brief Holds ragged segment bounds with an empty segment and a long one. */
            const std::vector<std::uint32_t> K_RAGGED_OFFSETS = {0, 3, 3, 10, 11, 1011};

            /** @brief Holds the ragged case inside wide buffers and alone. */
            struct WindowedSegments
            {
                static constexpr std::size_t K_VALUES_OFFSET = 5; /**< Values window start. */
                static constexpr std::size_t K_BOUNDS_OFFSET = 3; /**< Offsets window start. */
                static constexpr std::size_t K_OUTPUT_OFFSET = 4; /**< Output window start. */
                static constexpr std::size_t K_TAIL = 7;          /**< Elements past a window. */

                SushiRuntime::Buffer<float> wide_values;          /**< Values inside a window. */
                SushiRuntime::Buffer<std::uint32_t> wide_bounds;  /**< Offsets inside a window. */
                SushiRuntime::Buffer<float> wide_out;             /**< Output inside a window. */
                SushiRuntime::Buffer<float> values;               /**< Values alone. */
                SushiRuntime::Buffer<std::uint32_t> bounds;       /**< Offsets alone. */
                SushiRuntime::Buffer<float> out;                  /**< Output alone. */

                /** @brief Allocates both layouts and writes the same values into each. */
                explicit WindowedSegments(Runtime& runtime)
                    : wide_values(runtime.buffer<float>(K_VALUES_OFFSET + value_count() + K_TAIL)
                                      .value()),
                      wide_bounds(runtime.buffer<std::uint32_t>(
                                      K_BOUNDS_OFFSET + K_RAGGED_OFFSETS.size() + K_TAIL).value()),
                      wide_out(runtime.buffer<float>(K_OUTPUT_OFFSET + segments() + K_TAIL)
                                   .value()),
                      values(runtime.buffer<float>(value_count()).value()),
                      bounds(runtime.buffer<std::uint32_t>(K_RAGGED_OFFSETS.size()).value()),
                      out(runtime.buffer<float>(segments()).value())
                {
                    for (std::size_t i = 0; i < wide_values.size(); ++i)
                        wide_values[i] = 1.0e6f;
                    for (std::size_t i = 0; i < wide_bounds.size(); ++i)
                        wide_bounds[i] = 0;
                    for (std::size_t i = 0; i < wide_out.size(); ++i)
                        wide_out[i] = K_UNWRITTEN;
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
                [[nodiscard]] static std::size_t segments() { return K_RAGGED_OFFSETS.size() - 1; }

                /** @brief Returns the values the ragged case indexes. */
                [[nodiscard]] static std::size_t value_count() { return K_RAGGED_OFFSETS.back(); }

                /** @brief Returns the window that names the ragged case inside the wide buffers. */
                [[nodiscard]] static GraphPrimitives::SegmentedReduceWindow window()
                {
                    return GraphPrimitives::SegmentedReduceWindow{
                        K_VALUES_OFFSET, value_count(), K_BOUNDS_OFFSET, K_OUTPUT_OFFSET};
                }
            };

            /** @brief Records a segmented reduction of windows, fixed or late-bound. */
            SushiRuntime::Result<void>
            record_windowed(SushiRuntime::Recorder& recorder, const WindowedSegments& buffers,
                            bool late, const GraphPrimitives::SegmentedReduceWindow& window)
            {
                const std::size_t segments = WindowedSegments::segments();
                if (late)
                    return segmented_reduce(recorder, buffers.wide_values, buffers.wide_bounds,
                                            buffers.wide_out, LateCount::fixed(segments),
                                            Sum<float>{}, window);

                return segmented_reduce(recorder, buffers.wide_values, buffers.wide_bounds,
                                        buffers.wide_out, segments, Sum<float>{}, window);
            }
        } // namespace

        /** @brief Tests a segmented reduce over ragged segments, an empty one included. */
        TEST(Medium_GraphPrimitivesSegmentedReduce, FoldsRaggedSegmentsLeftToRight)
        {
            const std::vector<std::uint32_t>& offsets = K_RAGGED_OFFSETS;
            const std::size_t segments = offsets.size() - 1;
            const std::size_t value_count = offsets.back();

            auto runtime = Runtime::create().value();
            auto values = runtime.buffer<float>(value_count).value();
            auto bounds = runtime.buffer<std::uint32_t>(offsets.size()).value();
            auto out = runtime.buffer<float>(segments).value();

            std::vector<float> host(value_count);
            for (std::size_t i = 0; i < value_count; ++i)
            {
                host[i] = static_cast<float>(i % 13) * 0.25f;
                values[i] = host[i];
            }
            for (std::size_t i = 0; i < offsets.size(); ++i)
                bounds[i] = offsets[i];
            for (std::size_t s = 0; s < segments; ++s)
                out[s] = K_UNWRITTEN;

            auto graph = runtime.graph();
            segmented_reduce(graph.recorder(), values, bounds, out, segments, Sum<float>{}).value();
            graph.run().value();

            for (std::size_t s = 0; s < segments; ++s)
            {
                const float expected = host_segment_sum(host, offsets[s], offsets[s + 1]);
                EXPECT_TRUE(bit_equal(out[s], expected))
                    << "segment " << s << ": got " << out[s] << ", expected " << expected;
            }
        }

        /** @brief Tests that a live segment count below capacity leaves the rest unwritten. */
        TEST(Medium_GraphPrimitivesSegmentedReduce, LateBoundCountFoldsOnlyTheLiveSegments)
        {
            const std::vector<std::uint32_t>& offsets = K_RAGGED_OFFSETS;
            const std::size_t segments = offsets.size() - 1;
            const std::size_t value_count = offsets.back();

            auto runtime = Runtime::create().value();
            auto values = runtime.buffer<float>(value_count).value();
            auto bounds = runtime.buffer<std::uint32_t>(offsets.size()).value();
            auto out = runtime.buffer<float>(segments).value();

            std::vector<float> host(value_count);
            for (std::size_t i = 0; i < value_count; ++i)
            {
                host[i] = static_cast<float>(i % 13) * 0.25f;
                values[i] = host[i];
            }
            for (std::size_t i = 0; i < offsets.size(); ++i)
                bounds[i] = offsets[i];

            std::size_t live_segments = 0;
            const LateCount count =
                LateCount::from_host(segments, [&live_segments]() { return live_segments; });

            auto graph = runtime.graph();
            segmented_reduce(graph.recorder(), values, bounds, out, count, Sum<float>{}).value();

            for (std::size_t live : {std::size_t{2}, segments, std::size_t{0}, std::size_t{4}})
            {
                live_segments = live;
                for (std::size_t s = 0; s < segments; ++s)
                    out[s] = K_UNWRITTEN;

                graph.run().value();

                for (std::size_t s = 0; s < segments; ++s)
                {
                    const float expected = (s < live)
                        ? host_segment_sum(host, offsets[s], offsets[s + 1])
                        : K_UNWRITTEN;
                    EXPECT_TRUE(bit_equal(out[s], expected))
                        << "live " << live << ", segment " << s << ": got " << out[s];
                }
            }
            EXPECT_EQ(graph.compile_count(), 1u);
        }

        /** @brief Tests that short offsets or a short output are refused when the call records. */
        TEST(Medium_GraphPrimitivesSegmentedReduce, RejectsMisSizedArguments)
        {
            auto runtime = Runtime::create().value();
            auto values = runtime.buffer<float>(16).value();
            auto out = runtime.buffer<float>(1).value();
            auto wide_out = runtime.buffer<float>(4).value();
            auto short_bounds = runtime.buffer<std::uint32_t>(2).value();
            auto wide_bounds = runtime.buffer<std::uint32_t>(5).value();

            auto graph = runtime.graph();
            const auto refused_code = [&](const SushiRuntime::Buffer<std::uint32_t>& bounds,
                                          const SushiRuntime::Buffer<float>& output,
                                          bool late)
            {
                const SushiRuntime::Result<void> recorded =
                    late ? segmented_reduce(graph.recorder(), values, bounds, output,
                                            LateCount::fixed(4), Sum<float>{})
                         : segmented_reduce(graph.recorder(), values, bounds, output,
                                            std::size_t{4}, Sum<float>{});
                if (recorded)
                    return std::optional<SushiRuntime::ErrorCode>();

                return std::optional<SushiRuntime::ErrorCode>(recorded.error().code);
            };

            for (const bool late : {false, true})
            {
                EXPECT_EQ(refused_code(short_bounds, wide_out, late),
                          SushiRuntime::ErrorCode::InvalidGraph)
                    << "two offsets cannot bound four segments; late = " << late;
                EXPECT_EQ(refused_code(wide_bounds, out, late),
                          SushiRuntime::ErrorCode::InvalidGraph)
                    << "one output element cannot hold four results; late = " << late;
                EXPECT_EQ(refused_code(short_bounds, out, late),
                          SushiRuntime::ErrorCode::InvalidGraph)
                    << "late = " << late;
            }
            EXPECT_EQ(graph.recorder().node_count(), 0u) << "a refused call must record nothing";
        }

        /** @brief Tests that the fixed-count node launches at segment_count, not a buffer size. */
        TEST(Medium_GraphPrimitivesSegmentedReduce, FixedCountWidthIsTheSegmentCount)
        {
            constexpr std::size_t K_SEGMENTS = 64;

            // The offsets and the output are padded by different amounts past the count.
            auto runtime = Runtime::create().value();
            auto offsets = runtime.buffer<std::uint32_t>(K_SEGMENTS + 9).value();
            for (std::size_t s = 0; s <= K_SEGMENTS; ++s)
                offsets[s] = static_cast<std::uint32_t>(s);
            auto values = runtime.buffer<std::uint32_t>(K_SEGMENTS).value();
            for (std::size_t i = 0; i < K_SEGMENTS; ++i)
                values[i] = 1u;
            auto out = runtime.buffer<std::uint32_t>(K_SEGMENTS + 8).value();

            auto graph = runtime.graph();
            graph.set_profiling(true);
            segmented_reduce(graph.recorder(), values, offsets, out, K_SEGMENTS,
                             Sum<std::uint32_t>{}).value();

            const auto report = graph.run().value();

            ASSERT_EQ(report.node_events.size(), 1u);
            EXPECT_EQ(report.node_events[0].launched_width,
                      static_cast<std::uint32_t>(K_SEGMENTS));
            EXPECT_EQ(report.node_events[0].live_width, static_cast<std::uint32_t>(K_SEGMENTS));
            for (std::size_t s = 0; s < K_SEGMENTS; ++s)
                EXPECT_EQ(out[s], 1u) << "segment " << s;
        }

        /** @brief Tests that the late-bound form records a resolver and a node of capacity. */
        TEST(Medium_GraphPrimitivesSegmentedReduce, LateBoundWidthsAreOneAndTheCapacity)
        {
            constexpr std::size_t K_SEGMENTS = 128;
            const std::vector<std::uint32_t> expected_widths = {
                1u, static_cast<std::uint32_t>(K_SEGMENTS)};

            // The offsets and the output are padded by different amounts past the capacity.
            auto runtime = Runtime::create().value();
            auto offsets = runtime.buffer<std::uint32_t>(K_SEGMENTS + 9).value();
            for (std::size_t s = 0; s <= K_SEGMENTS; ++s)
                offsets[s] = static_cast<std::uint32_t>(s);
            auto values = runtime.buffer<std::uint32_t>(K_SEGMENTS).value();
            for (std::size_t i = 0; i < K_SEGMENTS; ++i)
                values[i] = 1u;
            auto out = runtime.buffer<std::uint32_t>(K_SEGMENTS + 8).value();

            auto graph = runtime.graph();
            graph.set_profiling(true);
            segmented_reduce(graph.recorder(), values, offsets, out, LateCount::fixed(K_SEGMENTS),
                             Sum<std::uint32_t>{}).value();
            EXPECT_EQ(graph.recorder().rule_count(), 1u) << "both nodes of one call are one rule";

            const auto report = graph.run().value();

            ASSERT_EQ(report.node_events.size(), expected_widths.size());
            for (std::size_t i = 0; i < expected_widths.size(); ++i)
            {
                EXPECT_EQ(report.node_events[i].launched_width, expected_widths[i])
                    << "node " << i;
                EXPECT_EQ(report.node_events[i].live_width, expected_widths[i]) << "node " << i;
            }
            for (std::size_t s = 0; s < K_SEGMENTS; ++s)
                EXPECT_EQ(out[s], 1u) << "segment " << s;
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
        /** @brief Tests that a segmented reduce over windows equals one over compact buffers. */
        TEST(Medium_GraphPrimitivesSegmentedReduce, WindowMatchesACompactBufferExactly)
        {
            auto runtime = Runtime::create().value();
            const std::size_t segments = WindowedSegments::segments();

            for (const bool late : {false, true})
            {
                WindowedSegments buffers(runtime);
                auto graph = runtime.graph();
                record_windowed(graph.recorder(), buffers, late, WindowedSegments::window())
                    .value();
                segmented_reduce(graph.recorder(), buffers.values, buffers.bounds, buffers.out,
                                 segments, Sum<float>{}).value();
                graph.run().value();

                for (std::size_t s = 0; s < segments; ++s)
                    EXPECT_TRUE(bit_equal(buffers.wide_out[WindowedSegments::K_OUTPUT_OFFSET + s],
                                          buffers.out[s]))
                        << "late = " << late << ", segment " << s;
                EXPECT_TRUE(bit_equal(buffers.wide_out[WindowedSegments::K_OUTPUT_OFFSET - 1],
                                      K_UNWRITTEN))
                    << "the element before the output window was written; late = " << late;
                EXPECT_TRUE(bit_equal(
                    buffers.wide_out[WindowedSegments::K_OUTPUT_OFFSET + segments], K_UNWRITTEN))
                    << "the element after the output window was written; late = " << late;
            }
        }

        /** @brief Tests that a writer of an operand's buffer waits only when it hits the window. */
        TEST(Medium_GraphPrimitivesSegmentedReduce, WindowOrdersOnlyOverlappingWriters)
        {
            using WindowProbe::region_of;
            constexpr std::size_t K_VALUES = WindowedSegments::K_VALUES_OFFSET;
            constexpr std::size_t K_OUTPUT = WindowedSegments::K_OUTPUT_OFFSET;
            auto runtime = Runtime::create().value();
            const WindowedSegments buffers(runtime);
            const std::size_t values_end = K_VALUES + WindowedSegments::value_count();
            const std::size_t output_end = K_OUTPUT + WindowedSegments::segments();

            const auto predecessors = [&](bool late, const SushiRuntime::BufferRegion& written)
            {
                auto graph = runtime.graph();
                record_windowed(graph.recorder(), buffers, late, WindowedSegments::window())
                    .value();
                return WindowProbe::predecessors_of_write(graph, written);
            };

            for (const bool late : {false, true})
            {
                EXPECT_EQ(predecessors(late, region_of(buffers.wide_values, 0, K_VALUES)), 0u)
                    << "values before the window; late = " << late;
                EXPECT_EQ(predecessors(late, region_of(buffers.wide_values, values_end, 1)), 0u)
                    << "values after the window; late = " << late;
                EXPECT_GT(predecessors(late, region_of(buffers.wide_values, K_VALUES, 1)), 0u)
                    << "the first value of the window; late = " << late;
                EXPECT_EQ(predecessors(late, region_of(buffers.wide_bounds, 0, 3)), 0u)
                    << "offsets before the window; late = " << late;
                EXPECT_GT(predecessors(late, region_of(buffers.wide_bounds, 3, 1)), 0u)
                    << "the first offset of the window; late = " << late;
                EXPECT_EQ(predecessors(late, region_of(buffers.wide_out, output_end, 1)), 0u)
                    << "output after the window; late = " << late;
                EXPECT_GT(predecessors(late, region_of(buffers.wide_out, output_end - 1, 1)), 0u)
                    << "the last output of the window; late = " << late;
            }
        }

        /** @brief Tests that a window past its buffer's end is refused, naming the operand. */
        TEST(Medium_GraphPrimitivesSegmentedReduce, RejectsAWindowPastItsBuffer)
        {
            auto runtime = Runtime::create().value();
            const WindowedSegments buffers(runtime);
            const GraphPrimitives::SegmentedReduceWindow fits = WindowedSegments::window();

            GraphPrimitives::SegmentedReduceWindow long_values = fits;
            long_values.values_count =
                WindowedSegments::value_count() + WindowedSegments::K_TAIL + 1;
            GraphPrimitives::SegmentedReduceWindow late_values = fits;
            late_values.values_offset = buffers.wide_values.size() + 1;
            late_values.values_count = GraphPrimitives::SegmentedReduceWindow::K_TO_END;
            GraphPrimitives::SegmentedReduceWindow late_bounds = fits;
            late_bounds.offsets_offset =
                WindowedSegments::K_BOUNDS_OFFSET + WindowedSegments::K_TAIL + 1;
            GraphPrimitives::SegmentedReduceWindow late_output = fits;
            late_output.output_offset =
                WindowedSegments::K_OUTPUT_OFFSET + WindowedSegments::K_TAIL + 1;

            const struct
            {
                GraphPrimitives::SegmentedReduceWindow window;
                const char* operand;
            } cases[] = {{long_values, "values"},
                         {late_values, "values"},
                         {late_bounds, "segment_offsets"},
                         {late_output, "output"}};

            auto graph = runtime.graph();
            for (const bool late : {false, true})
            {
                for (const auto& refused : cases)
                {
                    const SushiRuntime::Result<void> recorded =
                        record_windowed(graph.recorder(), buffers, late, refused.window);
                    ASSERT_FALSE(recorded.has_value()) << refused.operand << "; late = " << late;
                    EXPECT_EQ(recorded.error().code, SushiRuntime::ErrorCode::InvalidGraph);
                    EXPECT_NE(recorded.error().message.find(refused.operand), std::string::npos)
                        << recorded.error().message;
                }
            }
            EXPECT_EQ(graph.recorder().node_count(), 0u) << "a refused call must record nothing";
        }

    } // namespace Test
} // namespace SushiBLAS
