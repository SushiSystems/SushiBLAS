/**************************************************************************/
/* benchmark_harness.hpp                                                  */
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
 * @file benchmark_harness.hpp
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
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/io/io.hpp>
#include <SushiBLAS/recording_graph/recording_graph.hpp>

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
            case ExecutionMode::Auto:
                return "Auto";
            case ExecutionMode::ForceSYCL:
                return "ForceSYCL";
            case ExecutionMode::ForceReference:
                return "ForceReference";
        }
        return "UNKNOWN";
    }

    /// @brief Returns the name of a storage layout.
    [[nodiscard]] inline const char* layout_name(Core::Layout layout) noexcept
    {
        switch (layout)
        {
            case Core::Layout::RowMajor:
                return "RowMajor";
            case Core::Layout::ColumnMajor:
                return "ColumnMajor";
        }
        return "UNKNOWN";
    }

    /**
     * @brief Prints the device a probe tensor lives on, the build, the mode a RecordingGraph
     *        records under unless a benchmark sets another, and the layout.
     */
    inline void print_environment(Engine& engine)
    {
        const Tensor probe = engine.create_tensor({1});
        const std::string device = probe.device().get_info<sycl::info::device::name>();
        std::printf("device:        %s\n", device.c_str());
        std::printf("build:         %s\n", build_configuration_name());
        const Device::Profile& profile = Device::profile_for(probe.device());
        std::printf("compute units: %zu\n", profile.compute_units);
        std::printf("cache bytes:   %zu\n", profile.cache_bytes);
        std::printf("blocking:      inner %zu, outer %zu, stream %zu, min operand %zu\n",
                    profile.blocking.inner_bytes, profile.blocking.outer_bytes,
                    profile.blocking.stream_bytes, profile.blocking.min_operand_bytes);
        std::printf("record mode:   %s\n", execution_mode_name(K_DEFAULT_RECORDING_MODE));
        std::printf("layout:        %s\n", layout_name(engine.layout()));
        std::printf("iterations:    %d warm-up, %d timed\n\n", K_WARMUP_ITERATIONS,
                    K_TIMED_ITERATIONS);
    }

    /// @brief Returns the name of a benchmarked data type.
    [[nodiscard]] inline const char* dtype_name(Core::DataType dtype)
    {
        switch (dtype)
        {
            case Core::DataType::Half:
                return "HALF";
            case Core::DataType::Float32:
                return "FLOAT32";
            case Core::DataType::Float64:
                return "FLOAT64";
            default:
                throw std::invalid_argument("benchmark dtype is not HALF, FLOAT32 or FLOAT64");
        }
    }

    /// @brief Returns the bytes one element of a benchmarked data type occupies.
    [[nodiscard]] inline std::size_t dtype_bytes(Core::DataType dtype)
    {
        switch (dtype)
        {
            case Core::DataType::Half:
                return sizeof(sycl::half);
            case Core::DataType::Float32:
                return sizeof(float);
            case Core::DataType::Float64:
                return sizeof(double);
            default:
                throw std::invalid_argument("benchmark dtype is not HALF, FLOAT32 or FLOAT64");
        }
    }

    /// @brief Returns whether the engine's device holds the aspect a dtype's record check asks.
    [[nodiscard]] inline bool device_supports(Engine& engine, Core::DataType dtype)
    {
        const Tensor probe = engine.create_tensor({1});
        const Device::Profile& profile = Device::profile_for(probe.device());
        if (dtype == Core::DataType::Half)
            return profile.supports_fp16;
        if (dtype == Core::DataType::Float64 || dtype == Core::DataType::Complex64)
            return profile.supports_fp64;
        return true;
    }

    /**
     * @brief Writes one value into every element of a HALF, FLOAT32 or FLOAT64 tensor.
     * @throws std::invalid_argument for any other dtype.
     */
    inline void fill_tensor(Tensor& tensor, double value)
    {
        const std::size_t count = static_cast<std::size_t>(tensor.num_elements);
        switch (tensor.dtype)
        {
            case Core::DataType::Half:
                SushiBLAS::IO().from_vector(
                    tensor, std::vector<sycl::half>(count, static_cast<sycl::half>(value)));
                return;
            case Core::DataType::Float32:
                SushiBLAS::IO().from_vector(tensor,
                                        std::vector<float>(count, static_cast<float>(value)));
                return;
            case Core::DataType::Float64:
                SushiBLAS::IO().from_vector(tensor, std::vector<double>(count, value));
                return;
            default:
                throw std::invalid_argument("benchmark dtype is not HALF, FLOAT32 or FLOAT64");
        }
    }

    /// @brief Holds the median and the minimum of a set of timed iterations, in seconds.
    struct Timing
    {
        double median_seconds; /**< Median wall time of one iteration. */
        double min_seconds;    /**< Shortest wall time of one iteration. */
    };

    /**
     * @brief Runs warm-up iterations of a body, then times each of the timed iterations.
     * @tparam Body Callable that runs one whole iteration and returns when its work is done.
     */
    template<typename Body>
    [[nodiscard]] Timing time_iterations(Body body)
    {
        using Clock = std::chrono::steady_clock;

        for (int i = 0; i < K_WARMUP_ITERATIONS; ++i)
            body();

        std::vector<double> seconds;
        seconds.reserve(static_cast<std::size_t>(K_TIMED_ITERATIONS));
        for (int i = 0; i < K_TIMED_ITERATIONS; ++i)
        {
            const Clock::time_point start = Clock::now();
            body();
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

    /**
     * @brief Times recording one operation on a graph the benchmark owns and running that graph.
     *
     * Each iteration records on a graph that is empty, runs it and leaves it empty, so the
     * recording, the plan build and the run are all inside the timing.
     * @tparam Record Callable that records one operation through graph.recorder().
     * @throws std::runtime_error when a run fails or the recording is refused.
     */
    template<typename Record>
    [[nodiscard]] Timing measure(RecordingGraph& graph, Record record)
    {
        return time_iterations([&]()
        {
            record();
            graph.run().value();
        });
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
