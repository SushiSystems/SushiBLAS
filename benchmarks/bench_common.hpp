/**************************************************************************/
/* bench_common.hpp                                                       */
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
 * @file bench_common.hpp
 * @brief Declares the argument, environment, fill, timing and table helpers the benchmarks share.
 * @author Mustafa Garip
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <SushiBLAS/SushiBLAS.h>

/// @brief Holds the helpers shared by the benchmark programs.
namespace SushiBLAS::Bench
{
    /// @brief Number of untimed iterations run before the timed ones.
    constexpr int K_WARMUP_ITERATIONS = 2;

    /// @brief Number of timed iterations recorded per size and operation.
    constexpr int K_TIMED_ITERATIONS = 10;

    /// @brief Bytes in one gigabyte, as used for GB/s.
    constexpr double K_BYTES_PER_GIGABYTE = 1.0e9;

    /// @brief Floating-point operations in one GFLOP, as used for GFLOP/s.
    constexpr double K_FLOPS_PER_GIGAFLOP = 1.0e9;

    /**
     * @brief Returns the sizes given on the command line, or the defaults when none are given.
     * @throws std::invalid_argument when an argument is not a positive integer.
     */
    [[nodiscard]] inline std::vector<int64_t> parse_sizes(int argc, char** argv,
                                                          const std::vector<int64_t>& defaults)
    {
        if (argc <= 1)
            return defaults;

        std::vector<int64_t> sizes;
        sizes.reserve(static_cast<std::size_t>(argc - 1));
        for (int i = 1; i < argc; ++i)
        {
            char* end = nullptr;
            const long long value = std::strtoll(argv[i], &end, 10);
            if (end == argv[i] || *end != '\0' || value <= 0)
                throw std::invalid_argument(std::string("size is not a positive integer: ") +
                                            argv[i]);
            sizes.push_back(static_cast<int64_t>(value));
        }
        return sizes;
    }

    /// @brief Returns the name of the build configuration this program was compiled in.
    [[nodiscard]] inline const char* build_configuration_name() noexcept
    {
#ifdef NDEBUG
        return "release";
#else
        return "debug";
#endif
    }

    /// @brief Returns the name of an engine execution mode.
    [[nodiscard]] inline const char* execution_mode_name(ExecutionMode mode) noexcept
    {
        switch (mode)
        {
            case ExecutionMode::AUTO:
                return "AUTO";
            case ExecutionMode::FORCE_SYCL:
                return "FORCE_SYCL";
            case ExecutionMode::FORCE_REFERENCE:
                return "FORCE_REFERENCE";
        }
        return "UNKNOWN";
    }

    /// @brief Prints the device a probe tensor lives on, the build configuration and the engine mode.
    inline void print_environment(Engine& engine)
    {
        const Tensor probe = engine.create_tensor({1});
        const std::string device = probe.get_device().get_info<sycl::info::device::name>();
        std::printf("device:        %s\n", device.c_str());
        std::printf("build:         %s\n", build_configuration_name());
        std::printf("engine mode:   %s\n", execution_mode_name(engine.get_mode()));
        std::printf("iterations:    %d warm-up, %d timed\n\n", K_WARMUP_ITERATIONS,
                    K_TIMED_ITERATIONS);
    }

    /// @brief Returns the name of a benchmarked data type.
    [[nodiscard]] inline const char* dtype_name(Core::DataType dtype)
    {
        switch (dtype)
        {
            case Core::DataType::FLOAT32:
                return "FLOAT32";
            case Core::DataType::FLOAT64:
                return "FLOAT64";
            default:
                throw std::invalid_argument("benchmark dtype is neither FLOAT32 nor FLOAT64");
        }
    }

    /// @brief Returns the bytes one element of a benchmarked data type occupies.
    [[nodiscard]] inline std::size_t dtype_bytes(Core::DataType dtype)
    {
        switch (dtype)
        {
            case Core::DataType::FLOAT32:
                return sizeof(float);
            case Core::DataType::FLOAT64:
                return sizeof(double);
            default:
                throw std::invalid_argument("benchmark dtype is neither FLOAT32 nor FLOAT64");
        }
    }

    /**
     * @brief Writes one value into every element of a FLOAT32 or FLOAT64 tensor.
     * @throws std::invalid_argument for any other dtype.
     */
    inline void fill_tensor(Engine& engine, Tensor& tensor, double value)
    {
        const std::size_t count = static_cast<std::size_t>(tensor.num_elements);
        switch (tensor.dtype)
        {
            case Core::DataType::FLOAT32:
                SushiBLAS::IO(engine).from_vector(tensor,
                                        std::vector<float>(count, static_cast<float>(value)));
                return;
            case Core::DataType::FLOAT64:
                SushiBLAS::IO(engine).from_vector(tensor, std::vector<double>(count, value));
                return;
            default:
                throw std::invalid_argument("benchmark dtype is neither FLOAT32 nor FLOAT64");
        }
    }

    /// @brief Holds the median and the minimum of a set of timed iterations, in seconds.
    struct Timing
    {
        double median_seconds; /**< Median wall time of one iteration. */
        double min_seconds;    /**< Shortest wall time of one iteration. */
    };

    /**
     * @brief Runs warm-up iterations, then times each of the timed iterations.
     * @tparam Record Callable that records one operation on the engine's graph.
     */
    template<typename Record>
    [[nodiscard]] Timing measure(Engine& engine, Record record)
    {
        using Clock = std::chrono::steady_clock;

        for (int i = 0; i < K_WARMUP_ITERATIONS; ++i)
        {
            record();
            engine.execute();
        }

        std::vector<double> seconds;
        seconds.reserve(static_cast<std::size_t>(K_TIMED_ITERATIONS));
        for (int i = 0; i < K_TIMED_ITERATIONS; ++i)
        {
            const Clock::time_point start = Clock::now();
            record();
            engine.execute();
            const Clock::time_point stop = Clock::now();
            seconds.push_back(std::chrono::duration<double>(stop - start).count());
        }

        std::sort(seconds.begin(), seconds.end());
        const std::size_t middle = seconds.size() / 2;
        const double median = (seconds.size() % 2 == 0)
                                  ? 0.5 * (seconds[middle - 1] + seconds[middle])
                                  : seconds[middle];
        return Timing{median, seconds.front()};
    }

    /// @brief Prints benchmark results as a fixed-width table with one rate column.
    class ResultTable final
    {
        public:
            /// @brief Prints the header row naming the rate unit, such as "GB/s".
            explicit ResultTable(const char* rate_unit)
            {
                std::printf("%-10s %-8s %12s %12s %12s %10s\n", "op", "dtype", "size",
                            "median ms", "min ms", rate_unit);
            }

            /// @brief Prints one row; the rate is computed from the median time.
            void print_row(const char* op, Core::DataType dtype, int64_t size,
                           const Timing& timing, double work_per_iteration,
                           double work_per_unit) const
            {
                constexpr double K_MILLISECONDS_PER_SECOND = 1.0e3;
                const double rate = work_per_iteration / work_per_unit / timing.median_seconds;
                std::printf("%-10s %-8s %12lld %12.3f %12.3f %10.2f\n", op, dtype_name(dtype),
                            static_cast<long long>(size),
                            timing.median_seconds * K_MILLISECONDS_PER_SECOND,
                            timing.min_seconds * K_MILLISECONDS_PER_SECOND, rate);
            }
    };

    /**
     * @brief Runs a benchmark body and turns any exception into a message and exit code 1.
     * @tparam Body Callable taking (int argc, char** argv) and returning nothing.
     */
    template<typename Body>
    [[nodiscard]] int run_guarded(Body body, int argc, char** argv)
    {
        try
        {
            body(argc, argv);
            return EXIT_SUCCESS;
        }
        catch (const std::exception& error)
        {
            std::fprintf(stderr, "benchmark failed: %s\n", error.what());
            return EXIT_FAILURE;
        }
    }
} // namespace SushiBLAS::Bench
