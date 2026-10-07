/**************************************************************************/
/* bench_gemm.cpp                                                         */
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
 * @file bench_gemm.cpp
 * @brief Times square GEMM in FLOAT32 and FLOAT64 and prints GFLOP/s per size.
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

    /// @brief Square matrix orders timed when no size is given.
    const std::vector<int64_t> K_DEFAULT_SIZES = {128, 512, 1024, 2048};

    /// @brief Data types timed for every size.
    const std::vector<DataType> K_DTYPES = {DataType::Float32, DataType::Float64};

    /// @brief Value written into every element of A.
    constexpr double K_A_VALUE = 1.0;

    /// @brief Value written into every element of B.
    constexpr double K_B_VALUE = 0.5;

    /// @brief Times C = A * B for n x n matrices and prints one row.
    void bench_gemm(SushiBLAS::Engine& engine, const Bench::ResultTable& table, DataType dtype,
                    int64_t n)
    {
        SushiBLAS::Tensor a = engine.create_tensor({n, n}, dtype);
        SushiBLAS::Tensor b = engine.create_tensor({n, n}, dtype);
        SushiBLAS::Tensor c = engine.create_tensor({n, n}, dtype);
        Bench::fill_tensor(engine, a, K_A_VALUE);
        Bench::fill_tensor(engine, b, K_B_VALUE);

        const Bench::Timing timing =
            Bench::measure(engine, [&]() { SushiBLAS::BLASOps(engine).gemm(a, b, c); });
        const double order = static_cast<double>(n);
        const double flops = 2.0 * order * order * order;
        table.print_row("gemm", dtype, n, timing, flops, Bench::K_FLOPS_PER_GIGAFLOP);
    }

    /// @brief Times every data type at every requested size.
    void run(int argc, char** argv)
    {
        const std::vector<int64_t> sizes = Bench::parse_sizes(argc, argv, K_DEFAULT_SIZES);

        SushiRuntime::Execution::RuntimeContext context;
        SushiBLAS::Engine engine(context, SushiBLAS::Core::Layout::ColumnMajor);
        Bench::print_environment(engine);

        const Bench::ResultTable table("GFLOP/s");
        for (const DataType dtype : K_DTYPES)
        {
            for (const int64_t n : sizes)
                bench_gemm(engine, table, dtype, n);
        }
    }
} // namespace

/// @brief Runs the GEMM benchmark; sizes are optional matrix orders.
int main(int argc, char** argv)
{
    return Bench::run_guarded(run, argc, argv);
}
