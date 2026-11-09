/**************************************************************************/
/* test_packed_product.cpp                                                */
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
 * @file test_packed_product.cpp
 * @brief Tests that the packed product hands every sum of C, and only those, to its store.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/packed_product_plan.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_seam_kernels/reference/level3/gemm.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/memory_store.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/packed_product.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/packed_product_harness/packed_product_harness.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the packed product on the host CPU queue. */
        class Medium_SYCLPackedProduct : public HostDeviceTest
        {
            protected:
                /** @brief The outcome of one run: C, the writes per element and the plan. */
                struct Run
                {
                    std::vector<double> c;
                    std::vector<unsigned> calls;
                    Kernels::PackedProductPlan plan;
                };

                static constexpr int64_t K_M = 29;
                static constexpr int64_t K_N = 31;
                static constexpr int64_t K_K = 17;

                /** @brief Runs the product of the K_M x K_K and K_K x K_N matrices of small integers
                 *         through a MemoryStore that wants the macro-blocks below wanted_rows. */
                Run run_product(int64_t wanted_rows)
                {
                    const std::vector<double> a = small_integers<double>(K_M * K_K, 1);
                    const std::vector<double> b = small_integers<double>(K_K * K_N, 2);
                    const SharedArray<double> device_a(queue(), a);
                    const SharedArray<double> device_b(queue(), b);
                    SharedArray<double> device_c(queue(), static_cast<std::size_t>(K_M * K_N));
                    SharedArray<unsigned> calls(queue(), static_cast<std::size_t>(K_M * K_N));
                    const Kernels::PackedProductPlan plan = Kernels::plan_packed_blocks<double>(
                        ragged_blocking<double>(), 4, K_M, K_N, K_K, 1);
                    SharedArray<std::byte> scratch(queue(), plan.scratch_bytes);
                    const Kernels::OperandAccess a_access = Kernels::operand_access(
                        Core::Layout::RowMajor, Core::Transpose::NoTrans, K_K);
                    const Kernels::OperandAccess b_access = Kernels::operand_access(
                        Core::Layout::RowMajor, Core::Transpose::NoTrans, K_N);

                    Kernels::SYCL::Detail::packed_product<double>(
                        queue(), plan, K_M, K_N, K_K, device_a.get(), a_access, 0, device_b.get(),
                        b_access, 0, device_c.get(), 0, scratch.get(), {},
                        Kernels::SYCL::Detail::MemoryStore<double>{K_N, calls.get(), wanted_rows})
                        .wait();

                    return Run{device_c.values(), calls.values(), plan};
                }

                /** @brief Returns the reference product of the same matrices. */
                static std::vector<double> reference_product()
                {
                    const std::vector<double> a = small_integers<double>(K_M * K_K, 1);
                    const std::vector<double> b = small_integers<double>(K_K * K_N, 2);
                    std::vector<double> expected(static_cast<std::size_t>(K_M * K_N), 0.0);

                    Kernels::Reference::gemm<double>(
                        Core::Layout::RowMajor, Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                        K_M, K_N, K_K, 1.0, a.data(), K_K, b.data(), K_N, 0.0, expected.data(), K_N);

                    return expected;
                }
        };

        TEST_F(Medium_SYCLPackedProduct, StoresEverySumOfCOnce)
        {
            const Run run = run_product(K_M);

            ASSERT_GT(run.plan.blocks, 0);
            EXPECT_EQ(run.c, reference_product());
            EXPECT_EQ(run.calls, std::vector<unsigned>(static_cast<std::size_t>(K_M * K_N), 1u));
        }

        TEST_F(Medium_SYCLPackedProduct, AMacroBlockTheStoreDoesNotWantIsNeverComputedOrStored)
        {
            const Run probe = run_product(K_M);
            ASSERT_LT(probe.plan.block_rows, K_M);
            const int64_t wanted_rows = probe.plan.block_rows;
            const std::vector<double> expected = reference_product();
            const std::size_t wanted = static_cast<std::size_t>(wanted_rows * K_N);

            const Run run = run_product(wanted_rows);

            EXPECT_EQ(std::vector<double>(run.c.begin(), run.c.begin() + wanted),
                      std::vector<double>(expected.begin(), expected.begin() + wanted));
            EXPECT_EQ(std::vector<double>(run.c.begin() + wanted, run.c.end()),
                      std::vector<double>(run.c.size() - wanted, 0.0));
            EXPECT_EQ(std::vector<unsigned>(run.calls.begin(), run.calls.begin() + wanted),
                      std::vector<unsigned>(wanted, 1u));
            EXPECT_EQ(std::accumulate(run.calls.begin() + wanted, run.calls.end(), 0u), 0u);
        }
    } // namespace Test
} // namespace SushiBLAS
