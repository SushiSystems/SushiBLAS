/**************************************************************************/
/* benchmark_gemm_tiles.cpp                                               */
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
 * @file benchmark_gemm_tiles.cpp
 * @brief Times every tile shape of the tiled GEMM on the device and prints the table entries.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_catalog.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_choice.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp>

#include <SushiBLAS/benchmark_harness/benchmark_harness.hpp>

namespace
{
    namespace Bench = SushiBLAS::Bench;
    namespace Device = SushiBLAS::Device;
    namespace Kernels = SushiBLAS::Kernels;
    using SushiBLAS::Core::DataType;

    /// @brief Alignment, in bytes, of every array this program allocates.
    constexpr std::size_t K_ALIGNMENT = 128;

    /// @brief Orders of the square products timed; they span both size classes on most GPUs.
    constexpr std::array<int64_t, 6> K_ORDERS = {128, 256, 512, 1024, 2048, 4096};

    /// @brief Holds the score a shape collects in one size class: its time over the best, summed.
    struct ClassScore
    {
        double sum = 0.0; /**< Sum over orders of median time divided by the best median. */
        int orders = 0;   /**< Orders of the class the shape ran at. */
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

    /// @brief Times one shape on one square product and returns its median seconds.
    template<typename T, Kernels::GemmTileRole Role>
    [[nodiscard]] double time_shape(sycl::queue& q, int64_t n, const T* a, const T* b, T* c)
    {
        using SushiBLAS::Core::Layout;
        using SushiBLAS::Core::Transpose;

        const Kernels::OperandAccess access =
            Kernels::operand_access(Layout::ColumnMajor, Transpose::NoTrans, n);
        const Bench::Timing timing = Bench::time_iterations([&]()
        {
            Kernels::SYCL::Detail::gemm_local_tiled<T, Kernels::GemmTile<T, Role>>(
                q, n, n, n, T(1), a, access, 0, b, access, 0, T(0), c, access, 0, 1, {},
                Kernels::GemmIdentityEpilogue{}).wait();
        });
        return timing.median_seconds;
    }

    /// @brief Times one shape by role, dispatching to its compiled kernel.
    template<typename T>
    [[nodiscard]] double time_role(Kernels::GemmTileRole role, sycl::queue& q, int64_t n,
                                   const T* a, const T* b, T* c)
    {
        switch (role)
        {
            case Kernels::GemmTileRole::Wide:
                return time_shape<T, Kernels::GemmTileRole::Wide>(q, n, a, b, c);
            case Kernels::GemmTileRole::Compact:
                return time_shape<T, Kernels::GemmTileRole::Compact>(q, n, a, b, c);
            case Kernels::GemmTileRole::Balanced:
                break;
        }
        return time_shape<T, Kernels::GemmTileRole::Balanced>(q, n, a, b, c);
    }

    /// @brief Times every admissible shape at every order and prints one table entry per class.
    template<typename T>
    void tune(sycl::queue& q, const Device::Profile& profile, DataType dtype)
    {
        constexpr std::size_t K_CLASSES = 2;
        const auto shapes = Kernels::gemm_tile_traits<T>();
        std::array<std::array<ClassScore, Kernels::K_GEMM_TILE_ROLE_COUNT>, K_CLASSES> scores{};

        for (const int64_t n : K_ORDERS)
        {
            const std::size_t elements = static_cast<std::size_t>(n * n);
            const DeviceArray<T> a(q, elements, T(1));
            const DeviceArray<T> b(q, elements, T(0.5));
            const DeviceArray<T> c(q, elements, T(0));

            const Kernels::GemmTileProduct product{n, n, n, 1, dtype};
            const Kernels::GemmSizeClass size_class =
                Kernels::gemm_size_class(profile, shapes[0], product);

            std::array<double, Kernels::K_GEMM_TILE_ROLE_COUNT> seconds{};
            double best = std::numeric_limits<double>::infinity();
            for (const Kernels::GemmTileTraits& shape : shapes)
            {
                seconds[static_cast<std::size_t>(shape.role)] = 0.0;
                if (!Kernels::admits_gemm_tile(profile, shape))
                    continue;

                const double median = time_role<T>(shape.role, q, n, a.get(), b.get(), c.get());
                seconds[static_cast<std::size_t>(shape.role)] = median;
                best = median < best ? median : best;

                const double order = static_cast<double>(n);
                std::printf("%-6s %-9s %5lld  %10.3f ms  %9.1f GFLOP/s\n",
                            Bench::dtype_name(dtype), Kernels::to_string(shape.role).data(),
                            static_cast<long long>(n), median * 1.0e3,
                            2.0 * order * order * order / median / Bench::K_FLOPS_PER_GIGAFLOP);
            }

            for (const Kernels::GemmTileTraits& shape : shapes)
            {
                const double median = seconds[static_cast<std::size_t>(shape.role)];
                if (median <= 0.0)
                    continue;
                ClassScore& score =
                    scores[static_cast<std::size_t>(size_class)][static_cast<std::size_t>(shape.role)];
                score.sum += median / best;
                ++score.orders;
            }
        }

        std::printf("\n");
        for (std::size_t cls = 0; cls < K_CLASSES; ++cls)
        {
            const auto& class_scores = scores[cls];
            int orders = 0;
            for (const ClassScore& score : class_scores)
                orders = score.orders > orders ? score.orders : orders;
            if (orders == 0)
                continue;

            // Only a shape that ran at every order of the class can win it.
            std::size_t winner = Kernels::K_GEMM_TILE_ROLE_COUNT;
            for (std::size_t role = 0; role < class_scores.size(); ++role)
                if (class_scores[role].orders == orders &&
                    (winner == Kernels::K_GEMM_TILE_ROLE_COUNT ||
                     class_scores[role].sum < class_scores[winner].sum))
                    winner = role;

            std::printf("Kernels::GemmTileEntry{Device::Architecture::%s, Core::DataType::%s, "
                        "Kernels::GemmSizeClass::%s, Kernels::GemmTileRole::%s},\n",
                        Device::to_string(profile.architecture),
                        SushiBLAS::Core::to_string(dtype),
                        cls == 0 ? "Small" : "Large",
                        Kernels::to_string(Kernels::K_GEMM_TILE_ROLES[winner]).data());
        }
        std::printf("\n");
    }

    /// @brief Prints the device and tunes every dtype it computes in.
    void run(int, char**)
    {
        SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
        SushiBLAS::Engine engine(runtime, SushiBLAS::Core::Layout::ColumnMajor);
        Bench::print_environment(engine);

        sycl::queue& q = runtime.advanced().queue(SushiRuntime::DeviceIndex{0});
        const Device::Profile& profile = Device::profile_for(q);
        std::printf("architecture:  %s, %zu compute units, %zu B local per unit\n\n",
                    Device::to_string(profile.architecture), profile.compute_units,
                    profile.local_mem_per_compute_unit);

        if (!profile.has_dedicated_local_memory)
        {
            std::printf("the device has no dedicated local memory; the tiled GEMM does not run\n");
            return;
        }

        tune<float>(q, profile, DataType::Float32);
        if (profile.supports_fp64)
            tune<double>(q, profile, DataType::Float64);
    }
} // namespace

/// @brief Runs the tile tuner; see tests/README.md.
int main(int argc, char** argv)
{
    return Bench::run_guarded(run, argc, argv);
}
