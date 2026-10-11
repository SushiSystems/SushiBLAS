/**************************************************************************/
/* benchmark_level1.cpp                                                   */
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
 * @file benchmark_level1.cpp
 * @brief Times axpy, dot, nrm2, asum and iamax and prints GB/s per vector length.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>

#include "benchmark_common.hpp"

namespace
{
    namespace Bench = SushiBLAS::Bench;
    using SushiBLAS::Core::DataType;

    /// @brief Vector lengths timed when no size is given: 2^16, 2^20 and 2^24.
    const std::vector<int64_t> K_DEFAULT_SIZES = {int64_t{1} << 16, int64_t{1} << 20,
                                                  int64_t{1} << 24};

    /// @brief Data types timed for every length.
    const std::vector<DataType> K_DTYPES = {DataType::Float32, DataType::Float64};

    /// @brief Value written into every element of x.
    constexpr double K_X_VALUE = 1.0;

    /// @brief Value written into every element of y.
    constexpr double K_Y_VALUE = 2.0;

    /// @brief Scale axpy applies to x.
    constexpr double K_ALPHA = 0.5;

    /// @brief Times the five level 1 routines at one length and prints one row each.
    void benchmark_length(SushiBLAS::Engine& engine, const Bench::ResultTable& table, DataType dtype,
                      int64_t n)
    {
        SushiBLAS::Tensor x = engine.create_tensor({n}, dtype);
        SushiBLAS::Tensor y = engine.create_tensor({n}, dtype);
        SushiBLAS::Tensor scalar = engine.create_tensor({1}, dtype);
        SushiBLAS::Tensor index = engine.create_tensor({1}, DataType::Int64);
        Bench::fill_tensor(engine, x, K_X_VALUE);
        Bench::fill_tensor(engine, y, K_Y_VALUE);

        const double vector_bytes =
            static_cast<double>(n) * static_cast<double>(Bench::dtype_bytes(dtype));
        const double gb = Bench::K_BYTES_PER_GIGABYTE;

        // Bytes count each input element read once and each output element written once.
        table.print_row("axpy", dtype, n,
                        Bench::measure(
                            engine, [&]() { SushiBLAS::BLASOps(engine).axpy(K_ALPHA, x, y); }),
                        3.0 * vector_bytes, gb);
        table.print_row("dot", dtype, n,
                        Bench::measure(
                            engine, [&]() { SushiBLAS::BLASOps(engine).dot(x, y, scalar); }),
                        2.0 * vector_bytes, gb);
        table.print_row("nrm2", dtype, n,
                        Bench::measure(
                            engine, [&]() { SushiBLAS::BLASOps(engine).nrm2(x, scalar); }),
                        vector_bytes, gb);
        table.print_row("asum", dtype, n,
                        Bench::measure(
                            engine, [&]() { SushiBLAS::BLASOps(engine).asum(x, scalar); }),
                        vector_bytes, gb);
        table.print_row("iamax", dtype, n,
                        Bench::measure(
                            engine, [&]() { SushiBLAS::BLASOps(engine).iamax(x, index); }),
                        vector_bytes, gb);
    }

    /// @brief Times every data type at every requested length.
    void run(int argc, char** argv)
    {
        const std::vector<int64_t> sizes = Bench::parse_sizes(argc, argv, K_DEFAULT_SIZES);

        SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
        SushiBLAS::Engine engine(runtime);
        Bench::print_environment(engine);

        const Bench::ResultTable table("GB/s");
        for (const DataType dtype : K_DTYPES)
        {
            for (const int64_t n : sizes)
                benchmark_length(engine, table, dtype, n);
        }
    }
} // namespace

/// @brief Runs the level 1 benchmark; sizes are optional vector lengths.
int main(int argc, char** argv)
{
    return Bench::run_guarded(run, argc, argv);
}
