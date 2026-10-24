/**************************************************************************/
/* benchmark_random.cpp                                                   */
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
 * @file benchmark_random.cpp
 * @brief Times the random draws, shuffle and orthogonal and prints Melem/s per length.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <SushiBLAS/SushiBLAS.hpp>

#include <SushiBLAS/benchmark_harness/benchmark_harness.hpp>

namespace
{
    namespace Bench = SushiBLAS::Bench;
    using SushiBLAS::Core::DataType;

    /// @brief Vector lengths timed when no size is given: 2^16, 2^20 and 2^24.
    const std::vector<int64_t> K_DEFAULT_SIZES = {int64_t{1} << 16, int64_t{1} << 20,
                                                  int64_t{1} << 24};

    /// @brief Data types timed for every length.
    const std::vector<DataType> K_DTYPES = {DataType::Float32, DataType::Float64};

    /// @brief Matrix orders orthogonal is timed at, after the lengths.
    const std::vector<int64_t> K_ORTHOGONAL_ORDERS = {64, 256, 1024};

    /// @brief Longest vector shuffle is timed on; its Fisher-Yates walk is serial.
    constexpr int64_t K_SHUFFLE_MAX_LENGTH = int64_t{1} << 20;

    /// @brief Seed every draw starts from.
    constexpr uint64_t K_SEED = 42;

    /// @brief Rate of the Poisson draws Knuth's method serves.
    constexpr double K_SMALL_LAMBDA = 4.0;

    /// @brief Rate of the Poisson draws PTRS serves.
    constexpr double K_LARGE_LAMBDA = 30.0;

    /// @brief Largest integer discrete_uniform draws, from zero.
    constexpr int32_t K_DISCRETE_MAX = 9;

    /// @brief Elements per unit of the printed rate.
    constexpr double K_ELEMENTS_PER_MEGAELEMENT = 1.0e6;

    /// @brief Times every draw on one vector length and prints one row each.
    void benchmark_length(SushiBLAS::Engine& engine, const Bench::ResultTable& table,
                          DataType dtype, int64_t n)
    {
        Bench::BenchmarkGraph graph(engine);
        SushiBLAS::TaskRecorder& recorder = graph.recorder();
        SushiBLAS::RandomOps random(recorder, engine.rng());
        random.set_seed(K_SEED);
        SushiBLAS::Tensor t = engine.create_tensor({n}, dtype);
        const double elements = static_cast<double>(n);
        const double unit = K_ELEMENTS_PER_MEGAELEMENT;

        table.print_row("uniform", dtype, n,
                        Bench::measure(graph, [&]() { random.uniform(t); }), elements, unit);
        table.print_row("normal", dtype, n,
                        Bench::measure(graph, [&]() { random.normal(t); }), elements, unit);
        table.print_row("lognormal", dtype, n,
                        Bench::measure(graph, [&]() { random.log_normal(t); }), elements, unit);
        table.print_row("exp", dtype, n,
                        Bench::measure(graph, [&]() { random.exponential(t); }), elements,
                        unit);
        table.print_row("poisson 4", dtype, n,
                        Bench::measure(graph, [&]() { random.poisson(t, K_SMALL_LAMBDA); }),
                        elements, unit);
        table.print_row("poisson 30", dtype, n,
                        Bench::measure(graph, [&]() { random.poisson(t, K_LARGE_LAMBDA); }),
                        elements, unit);
        table.print_row("discrete", dtype, n,
                        Bench::measure(graph, [&]()
                                       { random.discrete_uniform(t, 0, K_DISCRETE_MAX); }),
                        elements, unit);
        table.print_row("truncnorm", dtype, n,
                        Bench::measure(graph, [&]() { random.truncated_normal(t); }), elements,
                        unit);
        table.print_row("bernoulli", dtype, n,
                        Bench::measure(graph, [&]() { random.bernoulli(t); }), elements, unit);
        if (n <= K_SHUFFLE_MAX_LENGTH)
            table.print_row("shuffle", dtype, n,
                            Bench::measure(graph, [&]() { random.shuffle(t); }), elements,
                            unit);
    }

    /// @brief Times orthogonal on one n x n matrix and prints one row, sized by n.
    void benchmark_orthogonal(SushiBLAS::Engine& engine, const Bench::ResultTable& table,
                              DataType dtype, int64_t n)
    {
        Bench::BenchmarkGraph graph(engine);
        SushiBLAS::TaskRecorder& recorder = graph.recorder();
        SushiBLAS::RandomOps random(recorder, engine.rng());
        random.set_seed(K_SEED);
        SushiBLAS::Tensor t = engine.create_tensor({n, n}, dtype);
        table.print_row("orthogonal", dtype, n,
                        Bench::measure(graph, [&]() { random.orthogonal(t); }),
                        static_cast<double>(n) * static_cast<double>(n),
                        K_ELEMENTS_PER_MEGAELEMENT);
    }

    /// @brief Times every data type at every requested length, then orthogonal.
    void run(int argc, char** argv)
    {
        const std::vector<int64_t> sizes = Bench::parse_sizes(argc, argv, K_DEFAULT_SIZES);

        SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
        SushiBLAS::Engine engine(runtime);
        Bench::print_environment(engine);

        const Bench::ResultTable table("Melem/s");
        for (const DataType dtype : K_DTYPES)
        {
            for (const int64_t n : sizes)
                benchmark_length(engine, table, dtype, n);
        }

        for (const DataType dtype : K_DTYPES)
        {
            for (const int64_t n : K_ORTHOGONAL_ORDERS)
                benchmark_orthogonal(engine, table, dtype, n);
        }
    }
} // namespace

/// @brief Runs the random benchmark; sizes are optional vector lengths.
int main(int argc, char** argv)
{
    return Bench::run_guarded(run, argc, argv);
}
