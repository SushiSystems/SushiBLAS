/**************************************************************************/
/* benchmark_reductions.cpp                                               */
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
 * @file benchmark_reductions.cpp
 * @brief Times full-tensor sum, max and var and prints GB/s per element count.
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

    /// @brief Element counts timed when no size is given: 2^16, 2^20 and 2^24.
    const std::vector<int64_t> K_DEFAULT_SIZES = {int64_t{1} << 16, int64_t{1} << 20,
                                                  int64_t{1} << 24};

    /// @brief Data types timed for every count.
    const std::vector<DataType> K_DTYPES = {DataType::Float32, DataType::Float64};

    /// @brief Value written into every input element.
    constexpr double K_INPUT_VALUE = 1.5;

    /// @brief Times the three reductions at one element count and prints one row each.
    void benchmark_count(SushiBLAS::Engine& engine, const Bench::ResultTable& table, DataType dtype,
                     int64_t n)
    {
        SushiBLAS::Tensor input = engine.create_tensor({n}, dtype);
        SushiBLAS::Tensor result = engine.create_tensor({1}, dtype);
        Bench::fill_tensor(engine, input, K_INPUT_VALUE);

        // Bytes count the input read once; the one-element result is ignored.
        const double input_bytes =
            static_cast<double>(n) * static_cast<double>(Bench::dtype_bytes(dtype));
        const double gb = Bench::K_BYTES_PER_GIGABYTE;

        table.print_row("sum", dtype, n,
                        Bench::measure(
                            engine, [&]() { SushiBLAS::ReductionOps(engine).sum(input, result); }),
                        input_bytes, gb);
        table.print_row("max", dtype, n,
                        Bench::measure(
                            engine, [&]() { SushiBLAS::ReductionOps(engine).max(input, result); }),
                        input_bytes, gb);
        table.print_row("var", dtype, n,
                        Bench::measure(
                            engine, [&]() { SushiBLAS::ReductionOps(engine).var(input, result); }),
                        input_bytes, gb);
    }

    /// @brief Times every data type at every requested count.
    void run(int argc, char** argv)
    {
        const std::vector<int64_t> sizes = Bench::parse_sizes(argc, argv, K_DEFAULT_SIZES);

        SushiRuntime::Execution::RuntimeContext context;
        SushiBLAS::Engine engine(context);
        Bench::print_environment(engine);

        const Bench::ResultTable table("GB/s");
        for (const DataType dtype : K_DTYPES)
        {
            for (const int64_t n : sizes)
                benchmark_count(engine, table, dtype, n);
        }
    }
} // namespace

/// @brief Runs the reductions benchmark; sizes are optional element counts.
int main(int argc, char** argv)
{
    return Bench::run_guarded(run, argc, argv);
}
