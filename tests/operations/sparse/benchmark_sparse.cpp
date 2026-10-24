/**************************************************************************/
/* benchmark_sparse.cpp                                                   */
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
 * @file benchmark_sparse.cpp
 * @brief Times spmv, spsymv, spmm, sptrsv, axpyi and doti and prints GB/s per matrix order.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <SushiBLAS/SushiBLAS.hpp>

#include <SushiBLAS/benchmark_harness/benchmark_harness.hpp>

namespace
{
    namespace Bench = SushiBLAS::Bench;
    using SushiBLAS::SparseOps;
    using SushiBLAS::Core::DataType;
    using SushiBLAS::Core::Transpose;
    using SushiBLAS::Core::Uplo;

    /// @brief Matrix orders timed when no size is given: 2^14, 2^17 and 2^20.
    const std::vector<int64_t> K_DEFAULT_SIZES = {int64_t{1} << 14, int64_t{1} << 17,
                                                  int64_t{1} << 20};

    /// @brief Entries stored in each row of every timed matrix.
    constexpr int64_t K_ENTRIES_PER_ROW = 16;

    /// @brief Distance between two entries of one row of the general matrix; a prime.
    constexpr int64_t K_COLUMN_STEP = 7919;

    /// @brief Columns of the dense operands of spmm.
    constexpr int64_t K_DENSE_COLUMNS = 8;

    /// @brief One stored position of the sparse vector per this many dense positions.
    constexpr int64_t K_VECTOR_SPACING = 16;

    /// @brief Value written into every element of x and B.
    constexpr double K_X_VALUE = 1.0;

    /// @brief Value written into every element of y and C.
    constexpr double K_Y_VALUE = 2.0;

    /// @brief Scale axpyi applies to the sparse vector.
    constexpr double K_ALPHA = 0.5;

    /// @brief Holds the triplets of one timed matrix, zero-based, with INT32 indices.
    template<typename T>
    struct Triplets
    {
        std::vector<int32_t> rows;    /**< The row of each entry. */
        std::vector<int32_t> columns; /**< The column of each entry. */
        std::vector<T> values;        /**< The value of each entry. */
    };

    /// @brief Returns the triplets of an n x n matrix with K_ENTRIES_PER_ROW spread entries a row.
    template<typename T>
    [[nodiscard]] Triplets<T> spread_triplets(int64_t n)
    {
        Triplets<T> triplets;
        const std::size_t capacity = static_cast<std::size_t>(n * K_ENTRIES_PER_ROW);
        triplets.rows.reserve(capacity);
        triplets.columns.reserve(capacity);
        triplets.values.reserve(capacity);
        for (int64_t i = 0; i < n; ++i)
        {
            for (int64_t t = 0; t < K_ENTRIES_PER_ROW; ++t)
            {
                triplets.rows.push_back(static_cast<int32_t>(i));
                triplets.columns.push_back(static_cast<int32_t>((i + t * K_COLUMN_STEP) % n));
                triplets.values.push_back(static_cast<T>(1.0 / static_cast<double>(t + 1)));
            }
        }
        return triplets;
    }

    /// @brief Returns the triplets of an n x n lower band whose diagonal dominates its row.
    template<typename T>
    [[nodiscard]] Triplets<T> lower_band_triplets(int64_t n)
    {
        Triplets<T> triplets;
        const std::size_t capacity = static_cast<std::size_t>(n * K_ENTRIES_PER_ROW);
        triplets.rows.reserve(capacity);
        triplets.columns.reserve(capacity);
        triplets.values.reserve(capacity);
        for (int64_t i = 0; i < n; ++i)
        {
            for (int64_t t = 0; t < K_ENTRIES_PER_ROW && t <= i; ++t)
            {
                const double value = (t == 0) ? static_cast<double>(K_ENTRIES_PER_ROW)
                                              : 1.0 / static_cast<double>(t + 1);
                triplets.rows.push_back(static_cast<int32_t>(i));
                triplets.columns.push_back(static_cast<int32_t>(i - t));
                triplets.values.push_back(static_cast<T>(value));
            }
        }
        return triplets;
    }

    /// @brief Times every sparse routine at one matrix order and prints one row each.
    template<typename T>
    void benchmark_order(SushiBLAS::Engine& engine, const Bench::ResultTable& table, DataType dtype,
                     int64_t n)
    {
        const SushiBLAS::SparseBuilder builder(engine.tensor_factory());

        const Triplets<T> general = spread_triplets<T>(n);
        const SushiBLAS::SparseMatrix A =
            builder.matrix_from_triplets(n, n, general.rows, general.columns, general.values);
        const SushiBLAS::SparseMatrix At =
            builder.matrix_from_triplets(n, n, general.columns, general.rows, general.values);
        const Triplets<T> band = lower_band_triplets<T>(n);
        const SushiBLAS::SparseMatrix L =
            builder.matrix_from_triplets(n, n, band.rows, band.columns, band.values);

        std::vector<int32_t> positions;
        positions.reserve(static_cast<std::size_t>((n + K_VECTOR_SPACING - 1) / K_VECTOR_SPACING));
        for (int64_t p = 0; p < n; p += K_VECTOR_SPACING)
            positions.push_back(static_cast<int32_t>(p));
        const SushiBLAS::SparseVector sx =
            builder.vector_from_entries(n, positions, std::vector<T>(positions.size(), T(1)));

        SushiBLAS::Tensor x = engine.create_tensor({n}, dtype);
        SushiBLAS::Tensor y = engine.create_tensor({n}, dtype);
        SushiBLAS::Tensor B = engine.create_tensor({n, K_DENSE_COLUMNS}, dtype);
        SushiBLAS::Tensor C = engine.create_tensor({n, K_DENSE_COLUMNS}, dtype);
        SushiBLAS::Tensor scalar = engine.create_tensor({1}, dtype);
        Bench::fill_tensor(x, K_X_VALUE);
        Bench::fill_tensor(y, K_Y_VALUE);
        Bench::fill_tensor(B, K_X_VALUE);
        Bench::fill_tensor(C, K_Y_VALUE);

        // Bytes count each stored entry with its index and each dense element once.
        const double element = static_cast<double>(Bench::dtype_bytes(dtype));
        const double entry = element + static_cast<double>(sizeof(int32_t));
        const double vector_bytes = static_cast<double>(n) * element;
        const double product_bytes = static_cast<double>(A.nnz()) * entry + 2.0 * vector_bytes;
        const double dense_bytes = static_cast<double>(A.nnz()) * entry +
                                   2.0 * vector_bytes * static_cast<double>(K_DENSE_COLUMNS);
        const double solve_bytes = static_cast<double>(L.nnz()) * entry + 2.0 * vector_bytes;
        const double stored = static_cast<double>(sx.nnz());
        const double gb = Bench::K_BYTES_PER_GIGABYTE;

        table.print_row("spmv", dtype, n,
                        Bench::measure(engine, [&]() { SparseOps(engine).spmv(A, x, y); }),
                        product_bytes, gb);
        table.print_row("spmv_t", dtype, n,
                        Bench::measure(
                            engine, [&]() { SparseOps(engine).spmv(A, x, y, Transpose::Trans); }),
                        product_bytes, gb);
        table.print_row("spmv_at", dtype, n,
                        Bench::measure(engine, [&]() { SparseOps(engine).spmv(At, x, y); }),
                        product_bytes, gb);
        table.print_row("spsymv", dtype, n,
                        Bench::measure(
                            engine, [&]() { SparseOps(engine).spsymv(L, x, y, Uplo::Lower); }),
                        solve_bytes, gb);
        table.print_row("spmm", dtype, n,
                        Bench::measure(engine, [&]() { SparseOps(engine).spmm(A, B, C); }),
                        dense_bytes, gb);
        table.print_row("spmm_t", dtype, n,
                        Bench::measure(
                            engine, [&]() { SparseOps(engine).spmm(A, B, C, Transpose::Trans); }),
                        dense_bytes, gb);
        table.print_row("sptrsv", dtype, n,
                        Bench::measure(engine,
                                       [&]() { SparseOps(engine).sptrsv(L, y, Uplo::Lower); }),
                        solve_bytes, gb);
        table.print_row("sptrsv_t", dtype, n,
                        Bench::measure(engine,
                                       [&]()
                                       {
                                           SparseOps(engine).sptrsv(L, y, Uplo::Lower,
                                                                    Transpose::Trans);
                                       }),
                        solve_bytes, gb);
        table.print_row("axpyi", dtype, n,
                        Bench::measure(engine, [&]() { SparseOps(engine).axpyi(K_ALPHA, sx, y); }),
                        stored * (entry + 2.0 * element), gb);
        table.print_row("doti", dtype, n,
                        Bench::measure(engine, [&]() { SparseOps(engine).doti(sx, y, scalar); }),
                        stored * (entry + element), gb);
    }

    /// @brief Times FLOAT32 and FLOAT64 at every requested matrix order.
    void run(int argc, char** argv)
    {
        const std::vector<int64_t> sizes = Bench::parse_sizes(argc, argv, K_DEFAULT_SIZES);

        SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
        SushiBLAS::Engine engine(runtime);
        Bench::print_environment(engine);

        const Bench::ResultTable table("GB/s");
        for (const int64_t n : sizes)
            benchmark_order<float>(engine, table, DataType::Float32, n);
        for (const int64_t n : sizes)
            benchmark_order<double>(engine, table, DataType::Float64, n);
    }
} // namespace

/// @brief Runs the sparse benchmark; sizes are optional matrix orders.
int main(int argc, char** argv)
{
    return Bench::run_guarded(run, argc, argv);
}
