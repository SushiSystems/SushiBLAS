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
 * @see modules/foundation/graph_primitives/tests/README.md
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/graph/graph.hpp>
#include <SushiRuntime/api/runtime/runtime.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>
#include <SushiRuntime/core/error.hpp>

#include <SushiBLAS/graph_primitives/combiners.hpp>
#include <SushiBLAS/graph_primitives/segmented_reduce.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using GraphPrimitives::Sum;
            using GraphPrimitives::segmented_reduce;
            using SushiRuntime::API::LateCount;
            using SushiRuntime::API::Runtime;
            using SushiRuntime::Core::SushiException;

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
        } // namespace

        /** @brief Tests a segmented reduce over ragged segments, an empty one included. */
        TEST(Unit_GraphPrimitivesSegmentedReduce, FoldsRaggedSegmentsLeftToRight)
        {
            const std::vector<std::uint32_t>& offsets = K_RAGGED_OFFSETS;
            const std::size_t segments = offsets.size() - 1;
            const std::size_t value_count = offsets.back();

            auto runtime = Runtime::create();
            auto values = runtime.buffer<float>(value_count);
            auto bounds = runtime.buffer<std::uint32_t>(offsets.size());
            auto out = runtime.buffer<float>(segments);

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
            segmented_reduce(graph.recorder(), values, bounds, out, segments, Sum<float>{});
            graph.run();

            for (std::size_t s = 0; s < segments; ++s)
            {
                const float expected = host_segment_sum(host, offsets[s], offsets[s + 1]);
                EXPECT_TRUE(bit_equal(out[s], expected))
                    << "segment " << s << ": got " << out[s] << ", expected " << expected;
            }
        }

        /** @brief Tests that a live segment count below capacity leaves the rest unwritten. */
        TEST(Unit_GraphPrimitivesSegmentedReduce, LateBoundCountFoldsOnlyTheLiveSegments)
        {
            const std::vector<std::uint32_t>& offsets = K_RAGGED_OFFSETS;
            const std::size_t segments = offsets.size() - 1;
            const std::size_t value_count = offsets.back();

            auto runtime = Runtime::create();
            auto values = runtime.buffer<float>(value_count);
            auto bounds = runtime.buffer<std::uint32_t>(offsets.size());
            auto out = runtime.buffer<float>(segments);

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
            segmented_reduce(graph.recorder(), values, bounds, out, count, Sum<float>{});

            for (std::size_t live : {std::size_t{2}, segments, std::size_t{0}, std::size_t{4}})
            {
                live_segments = live;
                for (std::size_t s = 0; s < segments; ++s)
                    out[s] = K_UNWRITTEN;

                graph.run();

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
        TEST(Unit_GraphPrimitivesSegmentedReduce, RejectsMisSizedArguments)
        {
            auto runtime = Runtime::create();
            auto values = runtime.buffer<float>(16);
            auto out = runtime.buffer<float>(1);
            auto wide_out = runtime.buffer<float>(4);
            auto short_bounds = runtime.buffer<std::uint32_t>(2);
            auto wide_bounds = runtime.buffer<std::uint32_t>(5);

            auto graph = runtime.graph();
            const auto refused_code = [&](const SushiRuntime::API::Buffer<std::uint32_t>& bounds,
                                          const SushiRuntime::API::Buffer<float>& output,
                                          bool late)
            {
                try
                {
                    if (late)
                        segmented_reduce(graph.recorder(), values, bounds, output,
                                         LateCount::fixed(4), Sum<float>{});
                    else
                        segmented_reduce(graph.recorder(), values, bounds, output, std::size_t{4},
                                         Sum<float>{});
                }
                catch (const SushiException& refused)
                {
                    return refused.error().code;
                }
                return SushiRuntime::Core::Errc::ok;
            };

            for (const bool late : {false, true})
            {
                EXPECT_EQ(refused_code(short_bounds, wide_out, late),
                          SushiRuntime::Core::Errc::invalid_graph)
                    << "two offsets cannot bound four segments; late = " << late;
                EXPECT_EQ(refused_code(wide_bounds, out, late),
                          SushiRuntime::Core::Errc::invalid_graph)
                    << "one output element cannot hold four results; late = " << late;
                EXPECT_EQ(refused_code(short_bounds, out, late),
                          SushiRuntime::Core::Errc::invalid_graph)
                    << "late = " << late;
            }
            EXPECT_EQ(graph.recorder().node_count(), 0u) << "a refused call must record nothing";
        }

        /** @brief Tests that the fixed-count node launches at segment_count, not a buffer size. */
        TEST(Unit_GraphPrimitivesSegmentedReduce, FixedCountWidthIsTheSegmentCount)
        {
            constexpr std::size_t K_SEGMENTS = 64;

            // The offsets and the output are padded by different amounts past the count.
            auto runtime = Runtime::create();
            auto offsets = runtime.buffer<std::uint32_t>(K_SEGMENTS + 9);
            for (std::size_t s = 0; s <= K_SEGMENTS; ++s)
                offsets[s] = static_cast<std::uint32_t>(s);
            auto values = runtime.buffer<std::uint32_t>(K_SEGMENTS);
            for (std::size_t i = 0; i < K_SEGMENTS; ++i)
                values[i] = 1u;
            auto out = runtime.buffer<std::uint32_t>(K_SEGMENTS + 8);

            auto graph = runtime.graph();
            graph.set_profiling(true);
            segmented_reduce(graph.recorder(), values, offsets, out, K_SEGMENTS,
                             Sum<std::uint32_t>{});

            const auto report = graph.run();

            ASSERT_EQ(report.node_events.size(), 1u);
            EXPECT_EQ(report.node_events[0].launched_width,
                      static_cast<std::uint32_t>(K_SEGMENTS));
            EXPECT_EQ(report.node_events[0].live_width, static_cast<std::uint32_t>(K_SEGMENTS));
            for (std::size_t s = 0; s < K_SEGMENTS; ++s)
                EXPECT_EQ(out[s], 1u) << "segment " << s;
        }

        /** @brief Tests that the late-bound form records a resolver and a node of capacity. */
        TEST(Unit_GraphPrimitivesSegmentedReduce, LateBoundWidthsAreOneAndTheCapacity)
        {
            constexpr std::size_t K_SEGMENTS = 128;
            const std::vector<std::uint32_t> expected_widths = {
                1u, static_cast<std::uint32_t>(K_SEGMENTS)};

            // The offsets and the output are padded by different amounts past the capacity.
            auto runtime = Runtime::create();
            auto offsets = runtime.buffer<std::uint32_t>(K_SEGMENTS + 9);
            for (std::size_t s = 0; s <= K_SEGMENTS; ++s)
                offsets[s] = static_cast<std::uint32_t>(s);
            auto values = runtime.buffer<std::uint32_t>(K_SEGMENTS);
            for (std::size_t i = 0; i < K_SEGMENTS; ++i)
                values[i] = 1u;
            auto out = runtime.buffer<std::uint32_t>(K_SEGMENTS + 8);

            auto graph = runtime.graph();
            graph.set_profiling(true);
            segmented_reduce(graph.recorder(), values, offsets, out, LateCount::fixed(K_SEGMENTS),
                             Sum<std::uint32_t>{});
            EXPECT_EQ(graph.recorder().rule_count(), 1u) << "both nodes of one call are one rule";

            const auto report = graph.run();

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

    } // namespace Test
} // namespace SushiBLAS
