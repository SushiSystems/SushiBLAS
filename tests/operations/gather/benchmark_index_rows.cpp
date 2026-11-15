/**************************************************************************/
/* benchmark_index_rows.cpp                                               */
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
 * @file benchmark_index_rows.cpp
 * @brief Times RowIndexOps beside the sparse products it replaces and prints GB/s per table size.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/gather/gather.hpp>

#include <SushiBLAS/benchmark_harness/benchmark_harness.hpp>

namespace
{
    namespace Bench = SushiBLAS::Bench;
    using SushiBLAS::RowIndexOps;
    using SushiBLAS::SparseOps;
    using SushiBLAS::Core::DataType;
    using SushiBLAS::Core::Transpose;

    /// @brief Table sizes timed when none is given: V, and R and C with it.
    const std::vector<int64_t> K_DEFAULT_SIZES = {8192};

    /// @brief D, the width of the table and of the rows.
    constexpr int64_t K_WIDTH = 384;

    /// @brief Distance between two positions of the index; a prime, so the positions spread.
    constexpr int64_t K_POSITION_STEP = 7919;

    /// @brief Value written into every element of the table, the rows and the picked tensor.
    constexpr double K_VALUE = 0.5;

    /// @brief Returns the position row r selects below extent.
    [[nodiscard]] int32_t position_of(int64_t r, int64_t extent) noexcept
    {
        return static_cast<int32_t>((r * K_POSITION_STEP) % extent);
    }

    /// @brief Holds the triplets of a selection matrix with one stored one per row.
    struct Selection
    {
        std::vector<int32_t> rows;    /**< The row of each entry. */
        std::vector<int32_t> columns; /**< The column of each entry. */
    };

    /// @brief Returns the entries of an R by V matrix that selects position_of(r, V) in row r.
    [[nodiscard]] Selection row_selection(int64_t count, int64_t extent)
    {
        Selection selection;
        for (int64_t r = 0; r < count; ++r)
        {
            selection.rows.push_back(static_cast<int32_t>(r));
            selection.columns.push_back(position_of(r, extent));
        }
        return selection;
    }

    /// @brief Returns the entries of an R by R*C matrix that selects element position_of(r, C).
    [[nodiscard]] Selection element_selection(int64_t count, int64_t width)
    {
        Selection selection;
        for (int64_t r = 0; r < count; ++r)
        {
            selection.rows.push_back(static_cast<int32_t>(r));
            selection.columns.push_back(static_cast<int32_t>(r * width + position_of(r, width)));
        }
        return selection;
    }

    /// @brief Returns an INT32 tensor holding the positions of count rows below extent.
    [[nodiscard]] SushiBLAS::Tensor index_tensor(SushiBLAS::Engine& engine, int64_t count,
                                                 int64_t extent)
    {
        std::vector<int32_t> positions;
        for (int64_t r = 0; r < count; ++r)
            positions.push_back(position_of(r, extent));

        SushiBLAS::Tensor index = engine.create_tensor({count}, DataType::Int32);
        SushiBLAS::IO().from_vector(index, positions);
        return index;
    }

    /// @brief Times take_rows and add_rows beside the spmm and the transposed spmm of a selection.
    template<typename T>
    void benchmark_rows(SushiRuntime::Runtime& runtime, SushiBLAS::Engine& engine,
                        const Bench::ResultTable& table,
                        DataType dtype, int64_t extent)
    {
        SushiBLAS::RecordingGraph graph(runtime, engine.scratch_pool(), engine.mode());
        Bench::label_graph(graph, "index");
        SushiBLAS::TaskRecorder& recorder = graph.recorder();
        const SushiBLAS::SparseBuilder builder(engine.tensor_factory());

        const int64_t v = extent;
        const int64_t r = extent;
        const Selection selection = row_selection(r, v);
        const SushiBLAS::SparseMatrix selector = builder.matrix_from_triplets(
            r, v, selection.rows, selection.columns, std::vector<T>(selection.rows.size(), T(1)));

        SushiBLAS::Tensor source = engine.create_tensor({v, K_WIDTH}, dtype);
        SushiBLAS::Tensor rows = engine.create_tensor({r, K_WIDTH}, dtype);
        SushiBLAS::Tensor index = index_tensor(engine, r, v);
        Bench::fill_tensor(source, K_VALUE);
        Bench::fill_tensor(rows, K_VALUE);

        // A gather moves R rows and the index; a sum moves R rows into V.
        const double element = static_cast<double>(Bench::dtype_bytes(dtype));
        const double gather_bytes = 2.0 * static_cast<double>(r * K_WIDTH) * element +
                                    static_cast<double>(r) * sizeof(int32_t);
        const double sum_bytes = static_cast<double>((r + v) * K_WIDTH) * element +
                                 static_cast<double>(r) * sizeof(int32_t);
        const double gb = Bench::K_BYTES_PER_GIGABYTE;

        table.print_row("take_rows", dtype, extent,
                        Bench::measure(graph, [&]()
                                       { RowIndexOps(recorder).take_rows(source, index, rows); }),
                        gather_bytes, gb);
        table.print_row("spmm", dtype, extent,
                        Bench::measure(graph, [&]()
                                       { SparseOps(recorder).spmm(selector, source, rows); }),
                        gather_bytes, gb);
        table.print_row("add_rows", dtype, extent,
                        Bench::measure(graph, [&]()
                                       { RowIndexOps(recorder).add_rows(rows, index, source); }),
                        sum_bytes, gb);
        table.print_row("spmm_t", dtype, extent,
                        Bench::measure(graph,
                                       [&]()
                                       {
                                           SparseOps(recorder).spmm(selector, rows, source,
                                                                    Transpose::Trans);
                                       }),
                        sum_bytes, gb);
    }

    /// @brief Times put_along at C of extent beside the transposed spmm over R * C rows of width 1.
    template<typename T>
    void benchmark_along(SushiRuntime::Runtime& runtime, SushiBLAS::Engine& engine,
                         const Bench::ResultTable& table,
                         DataType dtype, int64_t extent)
    {
        SushiBLAS::RecordingGraph graph(runtime, engine.scratch_pool(), engine.mode());
        Bench::label_graph(graph, "index");
        SushiBLAS::TaskRecorder& recorder = graph.recorder();
        const SushiBLAS::SparseBuilder builder(engine.tensor_factory());

        const int64_t r = extent;
        const int64_t c = extent;
        const Selection selection = element_selection(r, c);
        const SushiBLAS::SparseMatrix selector = builder.matrix_from_triplets(
            r, r * c, selection.rows, selection.columns,
            std::vector<T>(selection.rows.size(), T(1)));

        SushiBLAS::Tensor picked = engine.create_tensor({r}, dtype);
        SushiBLAS::Tensor picked_column = engine.create_tensor({r, 1}, dtype);
        SushiBLAS::Tensor values = engine.create_tensor({r, c}, dtype);
        SushiBLAS::Tensor expanded = engine.create_tensor({r * c, 1}, dtype);
        SushiBLAS::Tensor index = index_tensor(engine, r, c);
        Bench::fill_tensor(picked, K_VALUE);
        Bench::fill_tensor(picked_column, K_VALUE);

        // Both forms write R * C elements and read R elements and the index.
        const double element = static_cast<double>(Bench::dtype_bytes(dtype));
        const double bytes = static_cast<double>(r * c) * element +
                             static_cast<double>(r) * (element + sizeof(int32_t));
        const double gb = Bench::K_BYTES_PER_GIGABYTE;

        table.print_row("put_along", dtype, extent,
                        Bench::measure(graph, [&]()
                                       { RowIndexOps(recorder).put_along(picked, index, values); }),
                        bytes, gb);
        table.print_row("spmm_t_w1", dtype, extent,
                        Bench::measure(graph,
                                       [&]()
                                       {
                                           SparseOps(recorder).spmm(selector, picked_column,
                                                                    expanded, Transpose::Trans);
                                       }),
                        bytes, gb);
    }

    /// @brief Times FLOAT32 and FLOAT64 at every requested table size.
    void run(int argc, char** argv)
    {
        const std::vector<int64_t> sizes = Bench::parse_sizes(argc, argv, K_DEFAULT_SIZES);

        SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
        SushiBLAS::Engine engine(runtime);
        Bench::print_environment(runtime, engine, "index_rows");
        std::printf("width:         %lld columns\n\n", static_cast<long long>(K_WIDTH));

        const Bench::ResultTable table("GB/s");
        for (const int64_t extent : sizes)
        {
            benchmark_rows<float>(runtime, engine, table, DataType::Float32, extent);
            benchmark_along<float>(runtime, engine, table, DataType::Float32, extent);
        }
        for (const int64_t extent : sizes)
        {
            benchmark_rows<double>(runtime, engine, table, DataType::Float64, extent);
            benchmark_along<double>(runtime, engine, table, DataType::Float64, extent);
        }
    }
} // namespace

/// @brief Runs the index rows benchmark; sizes are optional table sizes.
int main(int argc, char** argv)
{
    return Bench::run_guarded(run, argc, argv);
}
