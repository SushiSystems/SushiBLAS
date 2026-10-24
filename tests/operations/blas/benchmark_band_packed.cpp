/**************************************************************************/
/* benchmark_band_packed.cpp                                              */
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
 * @file benchmark_band_packed.cpp
 * @brief Times the band and packed Level 2 routines beside their dense siblings, in both layouts.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <cstdio>
#include <vector>
#include <SushiBLAS/SushiBLAS.hpp>

#include <SushiBLAS/benchmark_harness/benchmark_harness.hpp>

namespace
{
    namespace Bench = SushiBLAS::Bench;
    using SushiBLAS::Core::DataType;
    using SushiBLAS::Core::Layout;
    using SushiBLAS::Core::Uplo;

    /// @brief Matrix orders timed when no size is given.
    const std::vector<int64_t> K_DEFAULT_SIZES = {1024, 4096, 8192};

    /// @brief Data types timed for every order.
    const std::vector<DataType> K_DTYPES = {DataType::Float32, DataType::Float64};

    /// @brief Layouts every order is timed in, each on an engine of its own.
    const std::vector<Layout> K_LAYOUTS = {Layout::ColumnMajor, Layout::RowMajor};

    /// @brief Off-diagonals on each side of the diagonal in every band matrix timed.
    constexpr int64_t K_BAND = 8;

    /// @brief Value written into every element of every vector.
    constexpr double K_VECTOR_VALUE = 1.0;

    /// @brief Scale of the rank updates: zero, so the matrix they rewrite keeps its values.
    constexpr double K_UPDATE_SCALE = 0.0;

    /// @brief Matrix elements per unit of the printed rate.
    constexpr double K_ELEMENTS_PER_MEGAELEMENT = 1.0e6;

    /// @brief Creates a band tensor of n lines of stored elements, each along the contiguous axis.
    SushiBLAS::Tensor create_band(SushiBLAS::Engine& engine, DataType dtype, int64_t stored,
                                  int64_t n)
    {
        if (engine.layout() == Layout::ColumnMajor)
            return engine.create_tensor({stored, n}, dtype);
        return engine.create_tensor({n, stored}, dtype);
    }

    /// @brief Times the dense, band and packed forms of the Level 2 routines at one order.
    void benchmark_order(SushiBLAS::Engine& engine, const Bench::ResultTable& table,
                         DataType dtype, int64_t n)
    {
        Bench::BenchmarkGraph graph(engine);
        SushiBLAS::TaskRecorder& recorder = graph.recorder();
        SushiBLAS::BLASOps blas(recorder);

        const int64_t packed_length = n * (n + 1) / 2;
        SushiBLAS::Tensor dense = engine.create_tensor({n, n}, dtype);
        SushiBLAS::Tensor band = create_band(engine, dtype, 2 * K_BAND + 1, n);
        SushiBLAS::Tensor triangle_band = create_band(engine, dtype, K_BAND + 1, n);
        SushiBLAS::Tensor packed = engine.create_tensor({packed_length}, dtype);
        SushiBLAS::Tensor x = engine.create_tensor({n}, dtype);
        SushiBLAS::Tensor y = engine.create_tensor({n}, dtype);

        // Every matrix element is 1 / n; with a unit diagonal no result leaves the float range.
        const double matrix_value = 1.0 / static_cast<double>(n);
        Bench::fill_tensor(dense, matrix_value);
        Bench::fill_tensor(band, matrix_value);
        Bench::fill_tensor(triangle_band, matrix_value);
        Bench::fill_tensor(packed, matrix_value);
        Bench::fill_tensor(x, K_VECTOR_VALUE);
        Bench::fill_tensor(y, K_VECTOR_VALUE);

        const double order = static_cast<double>(n);
        const double dense_elements = order * order;
        const double triangle_elements = static_cast<double>(packed_length);
        const double band_elements = order * static_cast<double>(2 * K_BAND + 1);
        const double triangle_band_elements = order * static_cast<double>(K_BAND + 1);
        const double unit = K_ELEMENTS_PER_MEGAELEMENT;
        const auto no_trans = SushiBLAS::Core::Transpose::NoTrans;
        const auto trans = SushiBLAS::Core::Transpose::Trans;
        const auto unit_diag = SushiBLAS::Core::Diag::Unit;

        table.print_row("gemv", dtype, n,
                        Bench::measure(graph, [&]() { blas.gemv(dense, x, y); }),
                        dense_elements, unit);
        table.print_row("gemv T", dtype, n,
                        Bench::measure(graph, [&]() { blas.gemv(dense, x, y, trans); }),
                        dense_elements, unit);
        table.print_row("gbmv", dtype, n,
                        Bench::measure(graph, [&]() { blas.gbmv(band, x, y, K_BAND, K_BAND); }),
                        band_elements, unit);

        table.print_row("symv", dtype, n,
                        Bench::measure(graph, [&]() { blas.symv(dense, x, y, Uplo::Upper); }),
                        dense_elements, unit);
        table.print_row("sbmv", dtype, n,
                        Bench::measure(graph, [&]()
                                       { blas.sbmv(triangle_band, x, y, Uplo::Upper, K_BAND); }),
                        band_elements, unit);
        table.print_row("spmv", dtype, n,
                        Bench::measure(graph, [&]() { blas.spmv(packed, x, y, Uplo::Upper); }),
                        dense_elements, unit);

        table.print_row("trmv", dtype, n,
                        Bench::measure(graph, [&]()
                                       { blas.trmv(dense, x, Uplo::Upper, no_trans, unit_diag); }),
                        triangle_elements, unit);
        table.print_row("tbmv", dtype, n,
                        Bench::measure(graph, [&]()
                                       {
                                           blas.tbmv(triangle_band, x, Uplo::Upper, K_BAND,
                                                     no_trans, unit_diag);
                                       }),
                        triangle_band_elements, unit);
        table.print_row("tpmv", dtype, n,
                        Bench::measure(graph, [&]()
                                       { blas.tpmv(packed, x, Uplo::Upper, no_trans, unit_diag); }),
                        triangle_elements, unit);

        table.print_row("trsv", dtype, n,
                        Bench::measure(graph, [&]()
                                       { blas.trsv(dense, x, Uplo::Upper, no_trans, unit_diag); }),
                        triangle_elements, unit);
        table.print_row("tbsv", dtype, n,
                        Bench::measure(graph, [&]()
                                       {
                                           blas.tbsv(triangle_band, x, Uplo::Upper, K_BAND,
                                                     no_trans, unit_diag);
                                       }),
                        triangle_band_elements, unit);
        table.print_row("tpsv", dtype, n,
                        Bench::measure(graph, [&]()
                                       { blas.tpsv(packed, x, Uplo::Upper, no_trans, unit_diag); }),
                        triangle_elements, unit);

        table.print_row("ger", dtype, n,
                        Bench::measure(graph, [&]() { blas.ger(x, y, dense, K_UPDATE_SCALE); }),
                        dense_elements, unit);
        table.print_row("syr", dtype, n,
                        Bench::measure(graph, [&]()
                                       { blas.syr(y, dense, Uplo::Upper, K_UPDATE_SCALE); }),
                        triangle_elements, unit);
        table.print_row("syr2", dtype, n,
                        Bench::measure(graph, [&]()
                                       { blas.syr2(x, y, dense, Uplo::Upper, K_UPDATE_SCALE); }),
                        triangle_elements, unit);
        table.print_row("spr", dtype, n,
                        Bench::measure(graph, [&]()
                                       { blas.spr(y, packed, Uplo::Upper, K_UPDATE_SCALE); }),
                        triangle_elements, unit);
        table.print_row("spr2", dtype, n,
                        Bench::measure(graph, [&]()
                                       { blas.spr2(x, y, packed, Uplo::Upper, K_UPDATE_SCALE); }),
                        triangle_elements, unit);
    }

    /// @brief Times every data type at every requested order in one layout.
    void benchmark_layout(SushiRuntime::Runtime& runtime, Layout layout,
                          const std::vector<int64_t>& sizes)
    {
        SushiBLAS::Engine engine(runtime, layout);
        Bench::print_environment(engine);

        const Bench::ResultTable table("Melem/s");
        for (const DataType dtype : K_DTYPES)
        {
            for (const int64_t n : sizes)
                benchmark_order(engine, table, dtype, n);
        }
        std::printf("\n");
    }

    /// @brief Times every requested order in each layout.
    void run(int argc, char** argv)
    {
        const std::vector<int64_t> sizes = Bench::parse_sizes(argc, argv, K_DEFAULT_SIZES);

        SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
        for (const Layout layout : K_LAYOUTS)
            benchmark_layout(runtime, layout, sizes);
    }
} // namespace

/// @brief Runs the band and packed benchmark; sizes are optional matrix orders.
int main(int argc, char** argv)
{
    return Bench::run_guarded(run, argc, argv);
}
