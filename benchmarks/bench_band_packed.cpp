/**************************************************************************/
/* bench_band_packed.cpp                                                  */
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
 * @file bench_band_packed.cpp
 * @brief Times the band and packed Level 2 routines beside their dense siblings.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>

#include "bench_common.hpp"

namespace
{
    namespace Bench = SushiBLAS::Bench;
    using SushiBLAS::Core::DataType;
    using SushiBLAS::Core::Uplo;

    /// @brief Matrix orders timed when no size is given.
    const std::vector<int64_t> K_DEFAULT_SIZES = {1024, 4096, 8192};

    /// @brief Data types timed for every order.
    const std::vector<DataType> K_DTYPES = {DataType::Float32, DataType::Float64};

    /// @brief Off-diagonals on each side of the diagonal in every band matrix timed.
    constexpr int64_t K_BAND = 8;

    /// @brief Value written into every element of every vector.
    constexpr double K_VECTOR_VALUE = 1.0;

    /// @brief Scale of the rank-1 updates: zero, so the matrix they rewrite keeps its values.
    constexpr double K_UPDATE_SCALE = 0.0;

    /// @brief Matrix elements per unit of the printed rate.
    constexpr double K_ELEMENTS_PER_MEGAELEMENT = 1.0e6;

    /// @brief Times the dense, band and packed forms of five Level 2 routines at one order.
    void bench_order(SushiBLAS::Engine& engine, const Bench::ResultTable& table, DataType dtype,
                     int64_t n)
    {
        SushiBLAS::BLASOps blas(engine);

        const int64_t packed_length = n * (n + 1) / 2;
        SushiBLAS::Tensor dense = engine.create_tensor({n, n}, dtype);
        SushiBLAS::Tensor band = engine.create_tensor({2 * K_BAND + 1, n}, dtype);
        SushiBLAS::Tensor triangle_band = engine.create_tensor({K_BAND + 1, n}, dtype);
        SushiBLAS::Tensor packed = engine.create_tensor({packed_length}, dtype);
        SushiBLAS::Tensor x = engine.create_tensor({n}, dtype);
        SushiBLAS::Tensor y = engine.create_tensor({n}, dtype);

        // Every matrix element is 1 / n; with a unit diagonal no result leaves the float range.
        const double matrix_value = 1.0 / static_cast<double>(n);
        Bench::fill_tensor(engine, dense, matrix_value);
        Bench::fill_tensor(engine, band, matrix_value);
        Bench::fill_tensor(engine, triangle_band, matrix_value);
        Bench::fill_tensor(engine, packed, matrix_value);
        Bench::fill_tensor(engine, x, K_VECTOR_VALUE);
        Bench::fill_tensor(engine, y, K_VECTOR_VALUE);

        const double order = static_cast<double>(n);
        const double dense_elements = order * order;
        const double triangle_elements = static_cast<double>(packed_length);
        const double band_elements = order * static_cast<double>(2 * K_BAND + 1);
        const double triangle_band_elements = order * static_cast<double>(K_BAND + 1);
        const double unit = K_ELEMENTS_PER_MEGAELEMENT;
        const auto no_trans = SushiBLAS::Core::Transpose::NoTrans;
        const auto unit_diag = SushiBLAS::Core::Diag::Unit;

        table.print_row("gemv", dtype, n,
                        Bench::measure(engine, [&]() { blas.gemv(dense, x, y); }),
                        dense_elements, unit);
        table.print_row("gbmv", dtype, n,
                        Bench::measure(engine, [&]() { blas.gbmv(band, x, y, K_BAND, K_BAND); }),
                        band_elements, unit);

        table.print_row("symv", dtype, n,
                        Bench::measure(engine, [&]() { blas.symv(dense, x, y, Uplo::Upper); }),
                        dense_elements, unit);
        table.print_row("sbmv", dtype, n,
                        Bench::measure(engine, [&]()
                                       { blas.sbmv(triangle_band, x, y, Uplo::Upper, K_BAND); }),
                        band_elements, unit);
        table.print_row("spmv", dtype, n,
                        Bench::measure(engine, [&]() { blas.spmv(packed, x, y, Uplo::Upper); }),
                        dense_elements, unit);

        table.print_row("trmv", dtype, n,
                        Bench::measure(engine, [&]()
                                       { blas.trmv(dense, x, Uplo::Upper, no_trans, unit_diag); }),
                        triangle_elements, unit);
        table.print_row("tbmv", dtype, n,
                        Bench::measure(engine, [&]()
                                       {
                                           blas.tbmv(triangle_band, x, Uplo::Upper, K_BAND,
                                                     no_trans, unit_diag);
                                       }),
                        triangle_band_elements, unit);
        table.print_row("tpmv", dtype, n,
                        Bench::measure(engine, [&]()
                                       { blas.tpmv(packed, x, Uplo::Upper, no_trans, unit_diag); }),
                        triangle_elements, unit);

        table.print_row("trsv", dtype, n,
                        Bench::measure(engine, [&]()
                                       { blas.trsv(dense, x, Uplo::Upper, no_trans, unit_diag); }),
                        triangle_elements, unit);
        table.print_row("tbsv", dtype, n,
                        Bench::measure(engine, [&]()
                                       {
                                           blas.tbsv(triangle_band, x, Uplo::Upper, K_BAND,
                                                     no_trans, unit_diag);
                                       }),
                        triangle_band_elements, unit);
        table.print_row("tpsv", dtype, n,
                        Bench::measure(engine, [&]()
                                       { blas.tpsv(packed, x, Uplo::Upper, no_trans, unit_diag); }),
                        triangle_elements, unit);

        table.print_row("syr", dtype, n,
                        Bench::measure(engine, [&]()
                                       { blas.syr(y, dense, Uplo::Upper, K_UPDATE_SCALE); }),
                        triangle_elements, unit);
        table.print_row("spr", dtype, n,
                        Bench::measure(engine, [&]()
                                       { blas.spr(y, packed, Uplo::Upper, K_UPDATE_SCALE); }),
                        triangle_elements, unit);
    }

    /// @brief Times every data type at every requested order.
    void run(int argc, char** argv)
    {
        const std::vector<int64_t> sizes = Bench::parse_sizes(argc, argv, K_DEFAULT_SIZES);

        SushiRuntime::Execution::RuntimeContext context;
        SushiBLAS::Engine engine(context, SushiBLAS::Core::Layout::ColumnMajor);
        Bench::print_environment(engine);

        const Bench::ResultTable table("Melem/s");
        for (const DataType dtype : K_DTYPES)
        {
            for (const int64_t n : sizes)
                bench_order(engine, table, dtype, n);
        }
    }
} // namespace

/// @brief Runs the band and packed benchmark; sizes are optional matrix orders.
int main(int argc, char** argv)
{
    return Bench::run_guarded(run, argc, argv);
}
