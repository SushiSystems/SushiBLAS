/**************************************************************************/
/* graph_primitives_demo.cpp                                              */
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
 * @file graph_primitives_demo.cpp
 * @brief Records the four graph primitives on a SushiRuntime graph and checks each on the host.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <SushiRuntime/sushiruntime.hpp>

#include <SushiBLAS/graph_primitives/combiners.hpp>
#include <SushiBLAS/graph_primitives/counting_sort.hpp>
#include <SushiBLAS/graph_primitives/prefix_scan.hpp>
#include <SushiBLAS/graph_primitives/reduce.hpp>
#include <SushiBLAS/graph_primitives/segmented_reduce.hpp>

namespace
{
    using Words = std::vector<std::uint32_t>;
    using WordBuffer = SushiRuntime::Buffer<std::uint32_t>;

    /** @brief Returns a shared buffer of capacity elements that starts with values. */
    [[nodiscard]] WordBuffer buffer_of(const SushiRuntime::Runtime& runtime, std::size_t capacity,
                                       const Words& values = {})
    {
        WordBuffer buffer = runtime.buffer<std::uint32_t>(capacity).value();
        for (std::size_t i = 0; i < values.size(); ++i)
            buffer[i] = values[i];

        return buffer;
    }

    /** @brief Prints whether the buffer starts with expected, and returns it. */
    [[nodiscard]] bool holds(const char* primitive, const WordBuffer& buffer, const Words& expected)
    {
        bool equal = true;
        for (std::size_t i = 0; i < expected.size(); ++i)
            equal = equal && buffer[i] == expected[i];

        std::cout << primitive << ": " << (equal ? "ok" : "MISMATCH") << "\n";
        return equal;
    }
} // namespace

int main()
{
    namespace Primitives = SushiBLAS::GraphPrimitives;
    using SushiRuntime::LateCount;

    constexpr std::size_t K_VALUE_COUNT = 1000;
    constexpr std::size_t K_SEGMENT_COUNT = 3;
    constexpr std::size_t K_KEY_COUNT = 6;
    constexpr std::size_t K_KEY_RANGE = 3;

    SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();

    // values holds 1, 2, ..., 1000; the middle segment of the three is empty.
    Words host_values(K_VALUE_COUNT);
    for (std::size_t i = 0; i < K_VALUE_COUNT; ++i)
        host_values[i] = static_cast<std::uint32_t>(i + 1);

    const WordBuffer values = buffer_of(runtime, K_VALUE_COUNT, host_values);
    const WordBuffer segment_offsets = buffer_of(runtime, K_SEGMENT_COUNT + 1, {0, 10, 10, 1000});
    const WordBuffer sum = buffer_of(runtime, 1);
    const WordBuffer segment_sums = buffer_of(runtime, K_SEGMENT_COUNT);
    const WordBuffer bucket_sizes = buffer_of(runtime, 4, {3, 0, 2, 5});
    const WordBuffer bucket_starts = buffer_of(runtime, 5);
    const WordBuffer bucket_total = buffer_of(runtime, 1);
    const WordBuffer keys = buffer_of(runtime, K_KEY_COUNT, {2, 0, 2, 1, 0, 2});
    const WordBuffer order = buffer_of(runtime, K_KEY_COUNT);
    const WordBuffer key_starts = buffer_of(runtime, K_KEY_RANGE + 1);
    const WordBuffer key_total = buffer_of(runtime, 1);

    // A primitive that is asked for more segments than the offsets describe records nothing.
    SushiRuntime::Graph refused_graph = runtime.graph();
    const SushiRuntime::Result<void> refused = Primitives::segmented_reduce(
        refused_graph.recorder(), values, segment_offsets, segment_sums, K_SEGMENT_COUNT + 2,
        Primitives::Sum<std::uint32_t>{});
    const bool refused_as_invalid_graph =
        !refused && refused.error().code == SushiRuntime::ErrorCode::InvalidGraph;
    if (!refused)
        std::cout << "refused: " << SushiRuntime::to_string(refused.error().code) << ": "
                  << refused.error().message << "\n";

    SushiRuntime::Graph graph = runtime.graph();
    Primitives::reduce(graph.recorder(), values, sum, K_VALUE_COUNT,
                       Primitives::Sum<std::uint32_t>{}).value();
    Primitives::segmented_reduce(graph.recorder(), values, segment_offsets, segment_sums,
                                 K_SEGMENT_COUNT, Primitives::Sum<std::uint32_t>{}).value();
    Primitives::prefix_scan(graph.recorder(), bucket_sizes, bucket_starts, bucket_total,
                            LateCount::fixed(4)).value();
    Primitives::counting_sort(graph.recorder(), keys, order, key_starts, key_total,
                              LateCount::fixed(K_KEY_COUNT), LateCount::fixed(K_KEY_RANGE)).value();

    graph.run().value();

    std::uint32_t host_sum = 0;
    for (const std::uint32_t value : host_values)
        host_sum += value;

    bool correct = holds("reduce", sum, {host_sum});
    correct = holds("segmented_reduce", segment_sums, {55, 0, host_sum - 55}) && correct;
    correct = holds("prefix_scan", bucket_starts, {0, 3, 3, 5, 10}) && correct;
    correct = holds("prefix_scan total", bucket_total, {10}) && correct;
    correct = holds("counting_sort order", order, {1, 4, 3, 0, 2, 5}) && correct;
    correct = holds("counting_sort starts", key_starts, {0, 2, 3, 6}) && correct;
    correct = refused_as_invalid_graph && correct;

    std::cout << "graph_primitives_demo: " << (correct ? "ok" : "MISMATCH") << "\n";
    return correct ? 0 : 1;
}
