/**************************************************************************/
/* test_gemm_depth_sliced.cpp                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file test_gemm_depth_sliced.cpp
 * @brief Tests the depth-sliced GEMM and its fold on the host CPU queue.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/blas_seam_kernels/extent_index.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_epilogue.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_catalog.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_depth_slice_plan.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm_depth_sliced.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Rows of C: one whole 64-row tile and an edge.
            constexpr int64_t K_M = 70;

            /// @brief Columns of C: an edge tile alone.
            constexpr int64_t K_N = 45;

            /// @brief The summed extent: three slices of 48, the last one of 4.
            constexpr int64_t K_K = 100;

            /// @brief K-steps per slice, three whole K tiles.
            constexpr int64_t K_SLICE_DEPTH = 48;

            /// @brief Products in the batch.
            constexpr int64_t K_BATCH = 2;

            /** @brief Returns a small integer, so every sum below is exact in double. */
            [[nodiscard]] double pattern(int64_t i, int64_t salt)
            {
                return static_cast<double>((i * 7 + salt) % 5 - 2);
            }

            /** @brief Owns one shared allocation of doubles on a queue. */
            class SharedDoubles final
            {
                public:
                    /** @brief Allocates count doubles shared with the device of q. */
                    SharedDoubles(sycl::queue& q, std::size_t count)
                        : queue_(q), data_(sycl::malloc_shared<double>(count, q))
                    {

                    }

                    SharedDoubles(const SharedDoubles&) = delete;
                    SharedDoubles& operator=(const SharedDoubles&) = delete;
                    SharedDoubles(SharedDoubles&&) = delete;
                    SharedDoubles& operator=(SharedDoubles&&) = delete;

                    /** @brief Frees the allocation. */
                    ~SharedDoubles() { sycl::free(data_, queue_); }

                    /** @brief Returns the first element. */
                    [[nodiscard]] double* get() const noexcept { return data_; }

                private:
                    /// @brief The queue the allocation belongs to.
                    sycl::queue& queue_;
                    /// @brief The allocation.
                    double* data_;
            };

            /** @brief Fills count elements of p from the pattern of salt. */
            void fill_pattern(double* p, int64_t count, int64_t salt)
            {
                for (int64_t i = 0; i < count; ++i)
                    p[i] = pattern(i, salt);
            }

            /** @brief Returns `alpha * A * B + beta * C` of row-major batches by a triple loop. */
            [[nodiscard]] std::vector<double> naive_product(const double* a, const double* b,
                                                            const double* c, double alpha,
                                                            double beta)
            {
                std::vector<double> out(static_cast<std::size_t>(K_BATCH * K_M * K_N));
                for (int64_t e = 0; e < K_BATCH; ++e)
                    for (int64_t r = 0; r < K_M; ++r)
                        for (int64_t col = 0; col < K_N; ++col)
                        {
                            double sum = 0.0;
                            for (int64_t p = 0; p < K_K; ++p)
                                sum += a[(e * K_M + r) * K_K + p] * b[(e * K_K + p) * K_N + col];
                            const int64_t at = (e * K_M + r) * K_N + col;
                            out[static_cast<std::size_t>(at)] = alpha * sum + beta * c[at];
                        }
                return out;
            }
        } // namespace

        /** @brief Runs the depth-sliced GEMM on the host CPU queue. */
        class Medium_GemmDepthSliced : public HostDeviceTest
        {

        };

        TEST_F(Medium_GemmDepthSliced, ThreeSlicesFoldToTheRowMajorProductUnderAlphaAndBeta)
        {
            using Balanced = Kernels::GemmTile<double, Kernels::GemmTileRole::Balanced>;
            if (!Device::hosts_local_tile(Device::profile_for(queue()),
                                          {Balanced::local_mem_bytes(),
                                           static_cast<std::size_t>(Balanced::K_WG_ITEMS)}))
            {
                GTEST_SKIP() << "the CPU device cannot host the Balanced work-group";
            }

            const int64_t str_a = K_M * K_K;
            const int64_t str_b = K_K * K_N;
            const int64_t str_c = K_M * K_N;
            const int64_t slices = Kernels::Indexing::ceil_div(K_K, K_SLICE_DEPTH);
            const int64_t partials = slices * K_BATCH * str_c;
            SharedDoubles a(queue(), static_cast<std::size_t>(K_BATCH * str_a));
            SharedDoubles b(queue(), static_cast<std::size_t>(K_BATCH * str_b));
            SharedDoubles c(queue(), static_cast<std::size_t>(K_BATCH * str_c));
            SharedDoubles scratch(queue(), static_cast<std::size_t>(partials));
            fill_pattern(a.get(), K_BATCH * str_a, 1);
            fill_pattern(b.get(), K_BATCH * str_b, 3);
            fill_pattern(c.get(), K_BATCH * str_c, 2);
            const std::vector<double> want = naive_product(a.get(), b.get(), c.get(), 2.0, -1.0);
            const Kernels::GemmDepthSlicePlan plan{
                slices, K_SLICE_DEPTH, static_cast<std::size_t>(partials) * sizeof(double)};

            Kernels::SYCL::Detail::gemm_depth_sliced<double>(
                queue(), plan, K_M, K_N, K_K, 2.0, a.get(), Kernels::OperandAccess{K_K, 1, false},
                str_a, b.get(), Kernels::OperandAccess{K_N, 1, false}, str_b, -1.0, c.get(),
                Kernels::OperandAccess{K_N, 1, false}, str_c, K_BATCH, scratch.get(), {},
                Kernels::GemmIdentityEpilogue{}).wait();

            EXPECT_EQ(std::vector<double>(c.get(), c.get() + K_BATCH * str_c), want);
        }
    } // namespace Test
} // namespace SushiBLAS
