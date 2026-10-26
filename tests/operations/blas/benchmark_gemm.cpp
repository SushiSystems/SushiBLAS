/**************************************************************************/
/* benchmark_gemm.cpp                                                     */
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
 * @file benchmark_gemm.cpp
 * @brief Times GEMM in both layouts, FLOAT32, FLOAT64 and HALF, and prints GFLOP/s per size.
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

    /// @brief Square matrix orders timed when no size is given.
    const std::vector<int64_t> K_DEFAULT_SIZES = {128, 512, 1024, 2048};

    /// @brief Data types timed for every square size, each where the device holds its aspect.
    const std::vector<DataType> K_DTYPES = {DataType::Float32, DataType::Float64, DataType::Half};

    /// @brief Layouts every section is timed in, each on an engine of its own.
    const std::vector<Layout> K_LAYOUTS = {Layout::ColumnMajor, Layout::RowMajor};

    /// @brief Value written into every element of A.
    constexpr double K_A_VALUE = 1.0;

    /// @brief Value written into every element of B.
    constexpr double K_B_VALUE = 0.5;

    /// @brief Times C = A * B for n x n matrices and prints one row.
    void benchmark_gemm(SushiBLAS::Engine& engine, const Bench::ResultTable& table, DataType dtype,
                    int64_t n)
    {
        SushiBLAS::RecordingGraph graph(engine);
        Bench::label_graph(graph, "gemm");
        SushiBLAS::TaskRecorder& recorder = graph.recorder();
        SushiBLAS::Tensor a = engine.create_tensor({n, n}, dtype);
        SushiBLAS::Tensor b = engine.create_tensor({n, n}, dtype);
        SushiBLAS::Tensor c = engine.create_tensor({n, n}, dtype);
        Bench::fill_tensor(a, K_A_VALUE);
        Bench::fill_tensor(b, K_B_VALUE);

        const Bench::Timing timing =
            Bench::measure(graph, [&]() { SushiBLAS::BLASOps(recorder).gemm(a, b, c); });
        const double order = static_cast<double>(n);
        const double flops = 2.0 * order * order * order;
        table.print_row("gemm", dtype, n, timing, flops, Bench::K_FLOPS_PER_GIGAFLOP);
    }

    /// @brief Holds the extents of one product with a small result and a long K.
    struct DeepShape
    {
        int64_t m = 0; ///< Rows of C.
        int64_t n = 0; ///< Columns of C.
        int64_t k = 0; ///< Rows of A, stored k x m, and of B, stored k x n.
    };

    /// @brief Shapes of C = A^T * B timed after the square sizes.
    const std::vector<DeepShape> K_DEEP_SHAPES = {
        {32, 27, 739328}, {64, 288, 184832}, {128, 576, 46208}, {256, 1152, 11552},
        {512, 2304, 2888}};

    /// @brief Times C = A^T * B for one deep shape and prints one row, sized by k.
    void benchmark_deep_gemm(SushiBLAS::Engine& engine, const Bench::ResultTable& table,
                             const DeepShape& shape)
    {
        SushiBLAS::RecordingGraph graph(engine);
        Bench::label_graph(graph, "gemm");
        SushiBLAS::TaskRecorder& recorder = graph.recorder();
        const DataType dtype = DataType::Float32;
        SushiBLAS::Tensor a = engine.create_tensor({shape.k, shape.m}, dtype);
        SushiBLAS::Tensor b = engine.create_tensor({shape.k, shape.n}, dtype);
        SushiBLAS::Tensor c = engine.create_tensor({shape.m, shape.n}, dtype);
        Bench::fill_tensor(a, K_A_VALUE);
        Bench::fill_tensor(b, K_B_VALUE);

        // Under ColumnMajor, A^T and B hold K along consecutive addresses.
        const Bench::Timing timing = Bench::measure(graph, [&]()
        {
            SushiBLAS::BLASOps(recorder).gemm(a, b, c, SushiBLAS::Core::Transpose::Trans,
                                              SushiBLAS::Core::Transpose::NoTrans);
        });
        const double flops = 2.0 * static_cast<double>(shape.m) * static_cast<double>(shape.n) *
                             static_cast<double>(shape.k);
        table.print_row("gemm deep", dtype, shape.k, timing, flops, Bench::K_FLOPS_PER_GIGAFLOP);
    }

    /// @brief Shapes of C = A * B^T timed last; a training step of YOLOv4-tiny forms them.
    const std::vector<DeepShape> K_STRIDED_SHAPES = {
        {32, 27, 86528}, {64, 288, 21632}, {64, 576, 21632}, {128, 1152, 5408},
        {255, 256, 1352}, {256, 2304, 1352}};

    /// @brief Times C = A * B^T for one deep shape and prints one row, sized by k.
    void benchmark_strided_gemm(SushiBLAS::Engine& engine, const Bench::ResultTable& table,
                                const DeepShape& shape)
    {
        SushiBLAS::RecordingGraph graph(engine);
        Bench::label_graph(graph, "gemm");
        SushiBLAS::TaskRecorder& recorder = graph.recorder();
        const DataType dtype = DataType::Float32;
        SushiBLAS::Tensor a = engine.create_tensor({shape.m, shape.k}, dtype);
        SushiBLAS::Tensor b = engine.create_tensor({shape.n, shape.k}, dtype);
        SushiBLAS::Tensor c = engine.create_tensor({shape.m, shape.n}, dtype);
        Bench::fill_tensor(a, K_A_VALUE);
        Bench::fill_tensor(b, K_B_VALUE);

        // Under ColumnMajor, A and B^T hold one K-step along consecutive addresses.
        const Bench::Timing timing = Bench::measure(graph, [&]()
        {
            SushiBLAS::BLASOps(recorder).gemm(a, b, c, SushiBLAS::Core::Transpose::NoTrans,
                                              SushiBLAS::Core::Transpose::Trans);
        });
        const double flops = 2.0 * static_cast<double>(shape.m) * static_cast<double>(shape.n) *
                             static_cast<double>(shape.k);
        table.print_row("gemm strided", dtype, shape.k, timing, flops,
                        Bench::K_FLOPS_PER_GIGAFLOP);
    }

    /// @brief Times every section in one layout, under the environment lines that name it.
    void benchmark_layout(SushiRuntime::Runtime& runtime, Layout layout,
                          const std::vector<int64_t>& sizes)
    {
        SushiBLAS::Engine engine(runtime, layout);
        Bench::print_environment(engine, "gemm");

        const Bench::ResultTable table("GFLOP/s");
        for (const DataType dtype : K_DTYPES)
        {
            if (!Bench::device_supports(engine, dtype))
            {
                std::printf("%-10s %-8s skipped: the device lacks the dtype's aspect\n", "gemm",
                            Bench::dtype_name(dtype));
                continue;
            }
            for (const int64_t n : sizes)
                benchmark_gemm(engine, table, dtype, n);
        }

        for (const DeepShape& shape : K_DEEP_SHAPES)
            benchmark_deep_gemm(engine, table, shape);

        for (const DeepShape& shape : K_STRIDED_SHAPES)
            benchmark_strided_gemm(engine, table, shape);
        std::printf("\n");
    }

    /// @brief Times every data type at every requested size, in each layout.
    void run(int argc, char** argv)
    {
        const std::vector<int64_t> sizes = Bench::parse_sizes(argc, argv, K_DEFAULT_SIZES);

        SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
        for (const Layout layout : K_LAYOUTS)
            benchmark_layout(runtime, layout, sizes);
    }
} // namespace

/// @brief Runs the GEMM benchmark; sizes are optional matrix orders.
int main(int argc, char** argv)
{
    return Bench::run_guarded(run, argc, argv);
}
