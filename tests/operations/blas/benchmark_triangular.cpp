/**************************************************************************/
/* benchmark_triangular.cpp                                               */
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
 * @file benchmark_triangular.cpp
 * @brief Times the triangular and symmetric BLAS routines in both layouts, in GFLOP/s per size.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blas_kernels/blas.hpp>
#include <SushiBLAS/SushiBLAS.hpp>

#include <SushiBLAS/benchmark_harness/benchmark_harness.hpp>

namespace
{
    namespace Bench = SushiBLAS::Bench;
    namespace Kernels = SushiBLAS::Kernels;
    using SushiBLAS::Core::DataType;
    using SushiBLAS::Core::Diag;
    using SushiBLAS::Core::Layout;
    using SushiBLAS::Core::Side;
    using SushiBLAS::Core::Transpose;
    using SushiBLAS::Core::Uplo;
    using SushiBLAS::ExecutionMode;

    /// @brief Matrix orders timed when no size is given.
    const std::vector<int64_t> K_DEFAULT_SIZES = {256, 1024, 4096};

    /// @brief Data types timed for every size.
    const std::vector<DataType> K_DTYPES = {DataType::Float32, DataType::Float64};

    /// @brief Diagonal block widths the direct TRSV rows time below the matrix order.
    const std::vector<int64_t> K_BLOCK_WIDTHS = {32, 64, 128, 256, 512, 1024};

    /// @brief Layouts every size is timed in, each on an engine of its own.
    const std::vector<Layout> K_LAYOUTS = {Layout::ColumnMajor, Layout::RowMajor};

    /// @brief Right-hand-side columns of the narrow TRSM rows.
    constexpr int64_t K_NARROW_COLUMNS = 8;

    /// @brief Value written into every element of G, the full operand of GEMM, SYRK and SYR2K.
    constexpr double K_G_VALUE = 1.0;

    /// @brief Value written into every element of H, the second operand of SYR2K.
    constexpr double K_H_VALUE = 0.5;

    /// @brief Value written into every element of B and of x.
    constexpr double K_ONE = 1.0;

    /// @brief Sets a recorder's execution mode and restores the earlier one on destruction.
    class ModeScope final
    {
        public:
            /// @brief Keeps the recorder's mode and sets the given one; the recorder outlives this.
            ModeScope(SushiBLAS::TaskRecorder& recorder, ExecutionMode mode)
                : recorder_(recorder), restored_(recorder.mode())
            {
                recorder_.set_mode(mode);
            }

            /// @brief Restores the mode the recorder had at construction.
            ~ModeScope()
            {
                recorder_.set_mode(restored_);
            }

            ModeScope(const ModeScope&) = delete;
            ModeScope& operator=(const ModeScope&) = delete;
            ModeScope(ModeScope&&) = delete;
            ModeScope& operator=(ModeScope&&) = delete;

        private:
            /// @brief The recorder whose mode is set.
            SushiBLAS::TaskRecorder& recorder_;

            /// @brief The mode the recorder had at construction.
            ExecutionMode restored_;
    };

    /// @brief Returns n as a floating-point factor of a flop count.
    [[nodiscard]] double as_factor(int64_t n) noexcept
    {
        return static_cast<double>(n);
    }

    /// @brief Writes a lower triangle of 1.0 / n under a diagonal of ones into an n x n tensor.
    template<typename T>
    void fill_unit_lower_as(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor, int64_t n)
    {
        const std::size_t order = static_cast<std::size_t>(n);
        const T below = static_cast<T>(1.0 / static_cast<double>(n));
        const bool by_columns = engine.layout() == Layout::ColumnMajor;
        std::vector<T> values(order * order, T(0));
        for (std::size_t col = 0; col < order; ++col)
        {
            values[col * order + col] = T(1);
            for (std::size_t row = col + 1; row < order; ++row)
                values[by_columns ? col * order + row : row * order + col] = below;
        }
        SushiBLAS::IO().from_vector(tensor, values);
    }

    /**
     * @brief Returns an n x n lower-triangular tensor with ones on the diagonal and 1.0 / n below.
     * @throws std::invalid_argument for a dtype that is neither FLOAT32 nor FLOAT64.
     */
    [[nodiscard]] SushiBLAS::Tensor unit_lower(SushiBLAS::Engine& engine, DataType dtype, int64_t n)
    {
        SushiBLAS::Tensor tensor = engine.create_tensor({n, n}, dtype);
        switch (dtype)
        {
            case DataType::Float32:
                fill_unit_lower_as<float>(engine, tensor, n);
                return tensor;
            case DataType::Float64:
                fill_unit_lower_as<double>(engine, tensor, n);
                return tensor;
            default:
                throw std::invalid_argument("benchmark dtype is neither FLOAT32 nor FLOAT64");
        }
    }

    /// @brief Returns a rows x cols tensor holding one value in every element.
    [[nodiscard]] SushiBLAS::Tensor constant_matrix(SushiBLAS::Engine& engine, DataType dtype,
                                                    int64_t rows, int64_t cols, double value)
    {
        SushiBLAS::Tensor tensor = engine.create_tensor({rows, cols}, dtype);
        Bench::fill_tensor(tensor, value);
        return tensor;
    }

    /// @brief Returns a vector of n ones.
    [[nodiscard]] SushiBLAS::Tensor ones_vector(SushiBLAS::Engine& engine, DataType dtype,
                                                int64_t n)
    {
        SushiBLAS::Tensor tensor = engine.create_tensor({n}, dtype);
        Bench::fill_tensor(tensor, K_ONE);
        return tensor;
    }

    /// @brief Times the four matrix products on n x n operands and prints one row each.
    void benchmark_products(SushiBLAS::Engine& engine, const Bench::ResultTable& table,
                        DataType dtype, int64_t n, const SushiBLAS::Tensor& a)
    {
        SushiBLAS::RecordingGraph graph(engine);
        Bench::label_graph(graph, "triangular");
        SushiBLAS::TaskRecorder& recorder = graph.recorder();
        const SushiBLAS::Tensor g = constant_matrix(engine, dtype, n, n, K_G_VALUE);
        const SushiBLAS::Tensor h = constant_matrix(engine, dtype, n, n, K_H_VALUE);
        const SushiBLAS::Tensor b = constant_matrix(engine, dtype, n, n, K_ONE);
        SushiBLAS::Tensor c = engine.create_tensor({n, n}, dtype);
        const double cube = as_factor(n) * as_factor(n) * as_factor(n);

        const Bench::Timing gemm =
            Bench::measure(graph, [&]() { SushiBLAS::BLASOps(recorder).gemm(g, b, c); });
        table.print_row("gemm", dtype, n, gemm, 2.0 * cube, Bench::K_FLOPS_PER_GIGAFLOP);

        const Bench::Timing symm = Bench::measure(graph, [&]()
        {
            SushiBLAS::BLASOps(recorder).symm(a, b, c, Side::Left, Uplo::Lower);
        });
        table.print_row("symm", dtype, n, symm, 2.0 * cube, Bench::K_FLOPS_PER_GIGAFLOP);

        const Bench::Timing syrk = Bench::measure(graph, [&]()
        {
            SushiBLAS::BLASOps(recorder).syrk(g, c, Uplo::Lower);
        });
        table.print_row("syrk", dtype, n, syrk, cube, Bench::K_FLOPS_PER_GIGAFLOP);

        const Bench::Timing syr2k = Bench::measure(graph, [&]()
        {
            SushiBLAS::BLASOps(recorder).syr2k(g, h, c, Uplo::Lower);
        });
        table.print_row("syr2k", dtype, n, syr2k, 2.0 * cube, Bench::K_FLOPS_PER_GIGAFLOP);
    }

    /// @brief Times TRMV on a vector of ones under one execution mode and prints one row.
    void benchmark_trmv(SushiBLAS::Engine& engine, const Bench::ResultTable& table, DataType dtype,
                    int64_t n, const SushiBLAS::Tensor& a, ExecutionMode mode, const char* label)
    {
        SushiBLAS::RecordingGraph graph(engine);
        Bench::label_graph(graph, "triangular");
        SushiBLAS::TaskRecorder& recorder = graph.recorder();
        SushiBLAS::Tensor x = ones_vector(engine, dtype, n);
        const ModeScope scope(recorder, mode);
        const Bench::Timing timing = Bench::measure(graph, [&]()
        {
            SushiBLAS::BLASOps(recorder).trmv(a, x, Uplo::Lower);
        });
        table.print_row(label, dtype, n, timing, as_factor(n) * as_factor(n),
                        Bench::K_FLOPS_PER_GIGAFLOP);
    }

    /// @brief Times TRSV on a vector of ones under one execution mode and prints one row.
    void benchmark_trsv(SushiBLAS::Engine& engine, const Bench::ResultTable& table, DataType dtype,
                    int64_t n, const SushiBLAS::Tensor& a, ExecutionMode mode, const char* label)
    {
        SushiBLAS::RecordingGraph graph(engine);
        Bench::label_graph(graph, "triangular");
        SushiBLAS::TaskRecorder& recorder = graph.recorder();
        SushiBLAS::Tensor x = ones_vector(engine, dtype, n);
        const ModeScope scope(recorder, mode);
        const Bench::Timing timing = Bench::measure(graph, [&]()
        {
            SushiBLAS::BLASOps(recorder).trsv(a, x, Uplo::Lower);
        });
        table.print_row(label, dtype, n, timing, as_factor(n) * as_factor(n),
                        Bench::K_FLOPS_PER_GIGAFLOP);
    }

    /// @brief Times TRSM from the left on n x columns right-hand sides of ones and prints one row.
    void benchmark_trsm(SushiBLAS::Engine& engine, const Bench::ResultTable& table, DataType dtype,
                    int64_t n, const SushiBLAS::Tensor& a, int64_t columns, const char* label)
    {
        SushiBLAS::RecordingGraph graph(engine);
        Bench::label_graph(graph, "triangular");
        SushiBLAS::TaskRecorder& recorder = graph.recorder();
        SushiBLAS::Tensor b = constant_matrix(engine, dtype, n, columns, K_ONE);
        const Bench::Timing timing = Bench::measure(graph, [&]()
        {
            SushiBLAS::BLASOps(recorder).trsm(a, b, Side::Left, Uplo::Lower);
        });
        table.print_row(label, dtype, n, timing, as_factor(n) * as_factor(n) * as_factor(columns),
                        Bench::K_FLOPS_PER_GIGAFLOP);
    }

    /**
     * @brief Times the SYCL TRSM kernel under one block width, called on the tensors' pointers.
     * @tparam T Element type of a's dtype.
     * @param block Steps of one diagonal block; the kernel clamps it to [1, n].
     * @pre No graph holds a recorded task that touches a.
     */
    template<typename T>
    void benchmark_solve_kernel(sycl::queue& queue, SushiBLAS::Engine& engine,
                            const Bench::ResultTable& table, int64_t n,
                            const SushiBLAS::Tensor& a, int64_t columns, int64_t block,
                            const std::string& label)
    {
        SushiBLAS::Tensor b = constant_matrix(engine, a.dtype, n, columns, K_ONE);
        const T* const pa = a.data_as<T>();
        T* const pb = b.data_as<T>();
        const Kernels::SYCL::TriangularSolvePlan plan{block};
        const int64_t ldb = engine.layout() == Layout::ColumnMajor ? n : columns;

        // FLOAT32 and FLOAT64 accumulate in their own type, so the kernel reads no workspace.
        const Bench::Timing timing = Bench::time_iterations([&]()
        {
            Kernels::SYCL::trsm<T>(queue, plan, engine.layout(), Side::Left, Uplo::Lower,
                                   Transpose::NoTrans, Diag::NonUnit, n, columns, T(1), pa, n,
                                   0, pb, ldb, 0, 1, nullptr, {}).wait();
        });
        table.print_row(label.c_str(), a.dtype, n, timing,
                        as_factor(n) * as_factor(n) * as_factor(columns),
                        Bench::K_FLOPS_PER_GIGAFLOP);
    }

    /// @brief Times the SYCL TRSM kernel at every block width for one element type.
    template<typename T>
    void benchmark_solve_kernels_as(sycl::queue& queue, SushiBLAS::Engine& engine,
                                const Bench::ResultTable& table, int64_t n,
                                const SushiBLAS::Tensor& a)
    {
        // A width of n or more is the single block the b=n row times.
        for (const int64_t block : K_BLOCK_WIDTHS)
        {
            if (block < n)
                benchmark_solve_kernel<T>(queue, engine, table, n, a, 1, block,
                                      "trsv b=" + std::to_string(block));
        }
        benchmark_solve_kernel<T>(queue, engine, table, n, a, 1, n, "trsv b=n");
        benchmark_solve_kernel<T>(queue, engine, table, n, a, K_NARROW_COLUMNS, n, "trsm b=n r=8");
        benchmark_solve_kernel<T>(queue, engine, table, n, a, n, n, "trsm b=n r=n");
    }

    /**
     * @brief Times the SYCL TRSM kernel at every block width for the dtype of a.
     * @throws std::invalid_argument for a dtype that is neither FLOAT32 nor FLOAT64.
     */
    void benchmark_solve_kernels(sycl::queue& queue, SushiBLAS::Engine& engine,
                             const Bench::ResultTable& table, int64_t n,
                             const SushiBLAS::Tensor& a)
    {
        switch (a.dtype)
        {
            case DataType::Float32:
                benchmark_solve_kernels_as<float>(queue, engine, table, n, a);
                return;
            case DataType::Float64:
                benchmark_solve_kernels_as<double>(queue, engine, table, n, a);
                return;
            default:
                throw std::invalid_argument("benchmark dtype is neither FLOAT32 nor FLOAT64");
        }
    }

    /// @brief Times every row of the program for one data type and one order.
    void benchmark_size(sycl::queue& queue, SushiBLAS::Engine& engine,
                    const Bench::ResultTable& table, DataType dtype, int64_t n)
    {
        const SushiBLAS::Tensor a = unit_lower(engine, dtype, n);

        benchmark_products(engine, table, dtype, n, a);
        benchmark_trmv(engine, table, dtype, n, a, ExecutionMode::ForceSYCL, "trmv");
        benchmark_trmv(engine, table, dtype, n, a, ExecutionMode::ForceReference, "trmv ref");
        benchmark_trsv(engine, table, dtype, n, a, ExecutionMode::ForceSYCL, "trsv");
        benchmark_trsv(engine, table, dtype, n, a, ExecutionMode::ForceReference, "trsv ref");
        benchmark_trsm(engine, table, dtype, n, a, K_NARROW_COLUMNS, "trsm r=8");
        benchmark_trsm(engine, table, dtype, n, a, n, "trsm r=n");
        benchmark_solve_kernels(queue, engine, table, n, a);
    }

    /// @brief Times every data type at every requested size in one layout.
    void benchmark_layout(SushiRuntime::Runtime& runtime, Layout layout,
                          const std::vector<int64_t>& sizes)
    {
        SushiBLAS::Engine engine(runtime, layout);
        Bench::print_environment(engine, "triangular");

        sycl::queue& queue = runtime.advanced().queue(SushiRuntime::DeviceIndex{0});
        const Bench::ResultTable table("GFLOP/s");
        for (const DataType dtype : K_DTYPES)
        {
            for (const int64_t n : sizes)
                benchmark_size(queue, engine, table, dtype, n);
        }
        std::printf("\n");
    }

    /// @brief Times every requested size in each layout.
    void run(int argc, char** argv)
    {
        const std::vector<int64_t> sizes = Bench::parse_sizes(argc, argv, K_DEFAULT_SIZES);

        SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
        for (const Layout layout : K_LAYOUTS)
            benchmark_layout(runtime, layout, sizes);
    }
} // namespace

/// @brief Runs the triangular benchmark; sizes are optional matrix orders.
int main(int argc, char** argv)
{
    return Bench::run_guarded(run, argc, argv);
}
