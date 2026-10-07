/**************************************************************************/
/* bench_gemm_blocking.cpp                                                */
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
 * @file bench_gemm_blocking.cpp
 * @brief Times one square GEMM kernel call under a cache blocking given on the command line.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/support/device_policy.hpp>
#include <SushiBLAS/kernels/blas/level3/packed_product_plan.hpp>
#include <SushiBLAS/kernels/sycl/blas/level3/gemm_packed.hpp>

#include "bench_common.hpp"

namespace
{
    namespace Bench = SushiBLAS::Bench;
    namespace Device = SushiBLAS::Device;
    namespace Kernels = SushiBLAS::Kernels;
    using SushiBLAS::Core::DataType;

    /// @brief Alignment, in bytes, of every array this program allocates.
    constexpr std::size_t K_ALIGNMENT = 128;

    /// @brief Value written into every element of A.
    constexpr double K_A_VALUE = 1.0;

    /// @brief Value written into every element of B.
    constexpr double K_B_VALUE = 0.5;

    /// @brief Holds what the command line asks for.
    struct Request
    {
        DataType dtype = DataType::Float32; /**< FLOAT32 or FLOAT64. */
        int64_t n = 0;                      /**< Order of the three square matrices. */
        bool packed = true;                 /**< Whether the packed kernel is the one timed. */
        Device::Profile profile;            /**< The device profile with the given overrides. */
    };

    /// @brief Owns one aligned device USM array and frees it with `sycl::free`.
    template<typename T>
    class DeviceArray final
    {
        public:
            /// @brief Allocates count elements on the device of q and fills them with value.
            DeviceArray(sycl::queue& q, std::size_t count, T value)
                : queue_(q), data_(sycl::aligned_alloc_device<T>(K_ALIGNMENT, count, q))
            {
                if (data_ == nullptr)
                    throw std::bad_alloc();
                q.fill(data_, value, count).wait();
            }

            /// @brief Frees the array.
            ~DeviceArray()
            {
                sycl::free(data_, queue_);
            }

            DeviceArray(const DeviceArray&) = delete;
            DeviceArray& operator=(const DeviceArray&) = delete;
            DeviceArray(DeviceArray&&) = delete;
            DeviceArray& operator=(DeviceArray&&) = delete;

            /// @brief Returns the first element.
            [[nodiscard]] T* get() const noexcept
            {
                return data_;
            }

        private:
            /// @brief The queue whose context owns the array.
            sycl::queue& queue_;

            /// @brief The first element.
            T* data_;
    };

    /**
     * @brief Returns the value of a `key=number` argument.
     * @throws std::invalid_argument when the text after `=` is not a non-negative integer.
     */
    [[nodiscard]] std::size_t parse_count(const std::string& argument, std::size_t key_length)
    {
        const char* text = argument.c_str() + key_length;
        char* end = nullptr;
        const unsigned long long value = std::strtoull(text, &end, 10);
        if (end == text || *end != '\0' || *text == '-')
            throw std::invalid_argument("not a non-negative integer: " + argument);
        return static_cast<std::size_t>(value);
    }

    /**
     * @brief Reads the dtype, the order and the overrides from the command line.
     * @throws std::invalid_argument on a missing or unknown argument.
     */
    [[nodiscard]] Request parse_request(int argc, char** argv,
                                        const Device::Profile& device_profile)
    {
        if (argc < 3)
            throw std::invalid_argument(
                "usage: bench_gemm_blocking <f32|f64> <n> [kernel=packed|unpacked] [inner=BYTES] "
                "[outer=BYTES] [stream=BYTES] [workers=PER_UNIT] [units=COUNT]");

        Request request;
        request.profile = device_profile;

        const std::string dtype = argv[1];
        if (dtype == "f32")
            request.dtype = DataType::Float32;
        else if (dtype == "f64")
            request.dtype = DataType::Float64;
        else
            throw std::invalid_argument("dtype is neither f32 nor f64: " + dtype);

        request.n = static_cast<int64_t>(parse_count(argv[2], 0));
        if (request.n <= 0)
            throw std::invalid_argument("n is not a positive integer: " + std::string(argv[2]));

        for (int i = 3; i < argc; ++i)
        {
            const std::string argument = argv[i];
            if (argument == "kernel=packed")
                request.packed = true;
            else if (argument == "kernel=unpacked")
                request.packed = false;
            else if (argument.rfind("inner=", 0) == 0)
                request.profile.blocking.inner_bytes = parse_count(argument, 6);
            else if (argument.rfind("outer=", 0) == 0)
                request.profile.blocking.outer_bytes = parse_count(argument, 6);
            else if (argument.rfind("stream=", 0) == 0)
                request.profile.blocking.stream_bytes = parse_count(argument, 7);
            else if (argument.rfind("workers=", 0) == 0)
                request.profile.blocking.workers_per_compute_unit = parse_count(argument, 8);
            else if (argument.rfind("units=", 0) == 0)
                request.profile.compute_units = parse_count(argument, 6);
            else
                throw std::invalid_argument("unknown argument: " + argument);
        }
        return request;
    }

    /// @brief Times C = A * B for one element type and prints the plan and one result row.
    template<typename T>
    void bench(sycl::queue& q, const Request& request)
    {
        namespace Kd = Kernels::SYCL::Detail;
        using SushiBLAS::Core::Layout;
        using SushiBLAS::Core::Transpose;

        const int64_t n = request.n;
        const std::size_t elements = static_cast<std::size_t>(n * n);
        const DeviceArray<T> a(q, elements, static_cast<T>(K_A_VALUE));
        const DeviceArray<T> b(q, elements, static_cast<T>(K_B_VALUE));
        const DeviceArray<T> c(q, elements, T(0));

        const Kernels::PackedProductPlan plan = Kernels::plan_packed_blocks<T>(
            request.profile.blocking, request.profile.compute_units, n, n, n, 1);
        const DeviceArray<unsigned char> scratch(q, plan.scratch_bytes,
                                                 static_cast<unsigned char>(0));

        std::printf("plan:          block %lld x %lld, depth %lld, %lld blocks, %lld workers, "
                    "%zu B of scratch\n\n",
                    static_cast<long long>(plan.block_rows),
                    static_cast<long long>(plan.block_cols),
                    static_cast<long long>(plan.block_depth),
                    static_cast<long long>(plan.blocks),
                    static_cast<long long>(plan.workers), plan.scratch_bytes);

        const Kernels::OperandAccess access =
            Kernels::operand_access(Layout::ColumnMajor, Transpose::NoTrans, n);
        const Bench::Timing timing = Bench::time_iterations([&]()
        {
            if (request.packed)
                Kd::packed_gemm<T>(q, plan, n, n, n, T(1), a.get(), access, 0, b.get(), access,
                                   0, T(0), c.get(), access, 0, 1, scratch.get(), {},
                                   Kernels::GemmIdentityEpilogue{}).wait();
            else
                Kernels::SYCL::gemm<T>(q, Layout::ColumnMajor, Transpose::NoTrans,
                                       Transpose::NoTrans, n, n, n, T(1), a.get(), n, 0,
                                       b.get(), n, 0, T(0), c.get(), n, 0, 1, {}).wait();
        });

        const Bench::ResultTable table("GFLOP/s");
        const double order = static_cast<double>(n);
        table.print_row(request.packed ? "packed" : "unpacked", request.dtype, n, timing,
                        2.0 * order * order * order, Bench::K_FLOPS_PER_GIGAFLOP);
    }

    /// @brief Parses the command line, prints the profile and times the requested kernel.
    void run(int argc, char** argv)
    {
        SushiRuntime::Execution::RuntimeContext context;
        SushiBLAS::Engine engine(context, SushiBLAS::Core::Layout::ColumnMajor);
        Bench::print_environment(engine);

        sycl::queue& q = context.get_queue();
        const Request request = parse_request(argc, argv, Device::profile_for(q));

        const Device::CacheBlocking& blocking = request.profile.blocking;
        std::printf("as run:        inner %zu, outer %zu, stream %zu, %zu workers per unit, "
                    "%zu units\n",
                    blocking.inner_bytes, blocking.outer_bytes, blocking.stream_bytes,
                    blocking.workers_per_compute_unit, request.profile.compute_units);

        if (request.dtype == DataType::Float32)
            bench<float>(q, request);
        else
            bench<double>(q, request);
    }
} // namespace

/// @brief Runs the blocking benchmark; see benchmarks/README.md for the arguments.
int main(int argc, char** argv)
{
    return Bench::run_guarded(run, argc, argv);
}
