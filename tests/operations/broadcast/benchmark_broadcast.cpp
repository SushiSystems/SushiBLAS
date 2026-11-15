/**************************************************************************/
/* benchmark_broadcast.cpp                                                */
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
 * @file benchmark_broadcast.cpp
 * @brief Times BroadcastOps::mul beside channel_affine and beside a per-index launch that divides.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/broadcast/broadcast.hpp>

#include <SushiBLAS/benchmark_harness/benchmark_harness.hpp>

namespace
{
    namespace Bench = SushiBLAS::Bench;
    using SushiBLAS::Core::DataType;

    /** @brief Holds the extents of one timed matrix. */
    struct Matrix
    {
        /** @brief Holds the outer extent, the count of a leading vector. */
        int64_t rows;
        /** @brief Holds the inner extent, the count of a trailing vector. */
        int64_t columns;
    };

    /// @brief The matrices timed: one with few columns and one with many.
    const std::vector<Matrix> K_MATRICES = {{8192, 384}, {384, 65536}};

    /// @brief The data type timed.
    constexpr DataType K_DTYPE = DataType::Float32;

    /// @brief The value every element and every vector holds.
    constexpr double K_VALUE = 1.0;

    /// @brief Times `mul` with a leading and with a trailing vector on one matrix.
    void benchmark_broadcast(SushiRuntime::Runtime& runtime, SushiBLAS::Engine& engine,
                             const Bench::ResultTable& table,
                             const Matrix& matrix, double bytes_per_iteration)
    {
        SushiBLAS::RecordingGraph graph(runtime, engine.scratch_pool(), engine.mode());
        Bench::label_graph(graph, "broadcast");
        SushiBLAS::BroadcastOps broadcast(graph.recorder());
        SushiBLAS::Tensor a = engine.create_tensor({matrix.rows, matrix.columns}, K_DTYPE);
        SushiBLAS::Tensor c = engine.create_tensor({matrix.rows, matrix.columns}, K_DTYPE);
        SushiBLAS::Tensor leading = engine.create_tensor({matrix.rows}, K_DTYPE);
        SushiBLAS::Tensor trailing = engine.create_tensor({matrix.columns}, K_DTYPE);
        Bench::fill_tensor(a, K_VALUE);
        Bench::fill_tensor(leading, K_VALUE);
        Bench::fill_tensor(trailing, K_VALUE);

        const int64_t elements = matrix.rows * matrix.columns;
        table.print_row("mul lead", K_DTYPE, elements,
                        Bench::measure(graph,
                                       [&]() { broadcast.mul(a, leading, c,
                                                             SushiBLAS::BroadcastAxis::Leading); }),
                        bytes_per_iteration, Bench::K_BYTES_PER_GIGABYTE);
        table.print_row("mul trail", K_DTYPE, elements,
                        Bench::measure(graph,
                                       [&]() { broadcast.mul(a, trailing, c,
                                                             SushiBLAS::BroadcastAxis::Trailing); }),
                        bytes_per_iteration, Bench::K_BYTES_PER_GIGABYTE);
    }

    /// @brief Times `channel_affine` with a scale of ones and a shift of zeros on one matrix.
    void benchmark_channel_affine(SushiRuntime::Runtime& runtime, SushiBLAS::Engine& engine,
                                  const Bench::ResultTable& table,
                                  const Matrix& matrix, double bytes_per_iteration)
    {
        SushiBLAS::RecordingGraph graph(runtime, engine.scratch_pool(), engine.mode());
        Bench::label_graph(graph, "broadcast");
        SushiBLAS::SpatialOps spatial(graph.recorder());
        SushiBLAS::Tensor in = engine.create_tensor({1, 1, matrix.rows, matrix.columns}, K_DTYPE);
        SushiBLAS::Tensor out = engine.create_tensor({1, 1, matrix.rows, matrix.columns}, K_DTYPE);
        SushiBLAS::Tensor gamma = engine.create_tensor({matrix.columns}, K_DTYPE);
        SushiBLAS::Tensor beta = engine.create_tensor({matrix.columns}, K_DTYPE);
        Bench::fill_tensor(in, K_VALUE);
        Bench::fill_tensor(gamma, K_VALUE);
        Bench::fill_tensor(beta, 0.0);

        table.print_row("affine", K_DTYPE, matrix.rows * matrix.columns,
                        Bench::measure(graph, [&]() { spatial.channel_affine(in, gamma, beta, out); }),
                        bytes_per_iteration, Bench::K_BYTES_PER_GIGABYTE);
    }

    /// @brief Times the same product as one index per element, with the division the range avoids.
    void benchmark_indexed(SushiRuntime::Runtime& runtime, SushiBLAS::Engine& engine,
                           const Bench::ResultTable& table,
                           const Matrix& matrix, double bytes_per_iteration)
    {
        SushiBLAS::RecordingGraph graph(runtime, engine.scratch_pool(), engine.mode());
        Bench::label_graph(graph, "broadcast");
        SushiBLAS::IndexOps index(graph.recorder());
        SushiBLAS::Tensor a = engine.create_tensor({matrix.rows, matrix.columns}, K_DTYPE);
        SushiBLAS::Tensor c = engine.create_tensor({matrix.rows, matrix.columns}, K_DTYPE);
        SushiBLAS::Tensor leading = engine.create_tensor({matrix.rows}, K_DTYPE);
        SushiBLAS::Tensor trailing = engine.create_tensor({matrix.columns}, K_DTYPE);
        Bench::fill_tensor(a, K_VALUE);
        Bench::fill_tensor(leading, K_VALUE);
        Bench::fill_tensor(trailing, K_VALUE);

        const std::size_t inner = static_cast<std::size_t>(matrix.columns);
        const std::size_t elements = static_cast<std::size_t>(matrix.rows * matrix.columns);
        const SushiBLAS::OperationOptions options{"bench.broadcast_indexed",
                                                  SushiRuntime::Determinism::Bitwise};

        const auto by_row = [inner](std::size_t i, const auto* x, const auto* v, auto* out)
        {
            out[i] = x[i] * v[i / inner];
        };
        const auto by_column = [inner](std::size_t i, const auto* x, const auto* v, auto* out)
        {
            out[i] = x[i] * v[i % inner];
        };

        table.print_row("idx lead", K_DTYPE, static_cast<int64_t>(elements),
                        Bench::measure(graph, [&]()
                        {
                            index.for_each(
                                elements,
                                SushiBLAS::Dispatch::bind_writes(
                                    SushiBLAS::Dispatch::bind_reads(
                                        by_row, SushiBLAS::Dispatch::read_as_element(a),
                                        SushiBLAS::Dispatch::read_as_element(leading)),
                                    SushiBLAS::Dispatch::write_as_element(c)),
                                options);
                        }),
                        bytes_per_iteration, Bench::K_BYTES_PER_GIGABYTE);
        table.print_row("idx trail", K_DTYPE, static_cast<int64_t>(elements),
                        Bench::measure(graph, [&]()
                        {
                            index.for_each(
                                elements,
                                SushiBLAS::Dispatch::bind_writes(
                                    SushiBLAS::Dispatch::bind_reads(
                                        by_column, SushiBLAS::Dispatch::read_as_element(a),
                                        SushiBLAS::Dispatch::read_as_element(trailing)),
                                    SushiBLAS::Dispatch::write_as_element(c)),
                                options);
                        }),
                        bytes_per_iteration, Bench::K_BYTES_PER_GIGABYTE);
    }

    /// @brief Times every operation on every matrix.
    void run(int argc, char** argv)
    {
        static_cast<void>(argc);
        static_cast<void>(argv);

        SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
        SushiBLAS::Engine engine(runtime);
        Bench::print_environment(runtime, engine, "broadcast");

        const Bench::ResultTable table("GB/s");
        for (const Matrix& matrix : K_MATRICES)
        {
            // One read of the matrix and one write of it; the vector is a rounding error beside them.
            const double bytes = 2.0 * static_cast<double>(matrix.rows) *
                                 static_cast<double>(matrix.columns) *
                                 static_cast<double>(Bench::dtype_bytes(K_DTYPE));
            benchmark_broadcast(runtime, engine, table, matrix, bytes);
            benchmark_channel_affine(runtime, engine, table, matrix, bytes);
            benchmark_indexed(runtime, engine, table, matrix, bytes);
        }
    }
} // namespace

/// @brief Runs the broadcast benchmark; it takes no argument.
int main(int argc, char** argv)
{
    return Bench::run_guarded(run, argc, argv);
}
