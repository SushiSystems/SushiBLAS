/**************************************************************************/
/* benchmark_elementwise.cpp                                              */
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
 * @file benchmark_elementwise.cpp
 * @brief Times add, in-place exp and in-place relu and prints GB/s per element count.
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

    /// @brief Value written into every element of the first operand.
    constexpr double K_A_VALUE = -0.25;

    /// @brief Value written into every element of the second operand.
    constexpr double K_B_VALUE = 0.75;

    /// @brief Times the three element-wise operations at one count and prints one row each.
    void benchmark_count(SushiBLAS::Engine& engine, const Bench::ResultTable& table, DataType dtype,
                     int64_t n)
    {
        SushiBLAS::Tensor a = engine.create_tensor({n}, dtype);
        SushiBLAS::Tensor b = engine.create_tensor({n}, dtype);
        SushiBLAS::Tensor c = engine.create_tensor({n}, dtype);
        Bench::fill_tensor(engine, a, K_A_VALUE);
        Bench::fill_tensor(engine, b, K_B_VALUE);

        const double tensor_bytes =
            static_cast<double>(n) * static_cast<double>(Bench::dtype_bytes(dtype));
        const double gb = Bench::K_BYTES_PER_GIGABYTE;

        // add reads two tensors and writes one; exp and relu read and write the same tensor.
        table.print_row("add", dtype, n,
                        Bench::measure(
                            engine, [&]() { SushiBLAS::ElementwiseOps(engine).add(a, b, c); }),
                        3.0 * tensor_bytes, gb);
        table.print_row("exp", dtype, n,
                        Bench::measure(
                            engine, [&]() { SushiBLAS::ElementwiseOps(engine).exp(c); }),
                        2.0 * tensor_bytes, gb);
        table.print_row("relu", dtype, n,
                        Bench::measure(
                            engine, [&]() { SushiBLAS::NonLinearOps(engine).relu(a); }),
                        2.0 * tensor_bytes, gb);
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

/// @brief Runs the element-wise benchmark; sizes are optional element counts.
int main(int argc, char** argv)
{
    return Bench::run_guarded(run, argc, argv);
}
