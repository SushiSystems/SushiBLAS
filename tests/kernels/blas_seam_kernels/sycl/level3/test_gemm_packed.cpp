/**************************************************************************/
/* test_gemm_packed.cpp                                                   */
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
 * @file test_gemm_packed.cpp
 * @brief Tests the packed SYCL gemm under three blockings against the reference gemm.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/packed_product_plan.hpp>
#include <SushiBLAS/blas_seam_kernels/reference/level3/gemm.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm_packed.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/packed_product_harness/packed_product_harness.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the packed gemm on the host CPU queue. */
        class Device_SYCLGemmPacked : public HostDeviceTest
        {

        };

        TEST_F(Device_SYCLGemmPacked, EveryBlockingMatchesTheReference)
        {
            // The patterned operands are small integers, so every blocking computes exactly.
            const Device::CacheBlocking blockings[] = {smallest_blocking(),
                                                       ragged_blocking<double>(),
                                                       widest_blocking()};
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                for (const bool trans_a : {false, true})
                {
                    for (const Device::CacheBlocking& blocking : blockings)
                    {
                        SCOPED_TRACE(trans_a);
                        GemmProblem<double> problem(queue(), layout, transpose_of(trans_a),
                                                    Core::Transpose::NoTrans, 45, 38, 52, 2);
                        Kernels::PackedProductPlan plan;

                        const std::vector<double> packed =
                            problem.packed(blocking, 4, 3.0, 1.0, &plan);

                        ASSERT_GT(plan.blocks, 0);
                        EXPECT_EQ(packed, problem.expected(3.0, 1.0));
                    }
                }
            }
        }

        TEST_F(Device_SYCLGemmPacked, TheEntryPointWithScratchMatchesTheReference)
        {
            const int64_t m = 70;
            const int64_t n = 66;
            const int64_t k = 58;
            const std::vector<float> a = small_integers<float>(m * k, 1);
            const std::vector<float> b = small_integers<float>(k * n, 2);
            std::vector<float> c = small_integers<float>(m * n, 3);
            const SharedArray<float> device_a(queue(), a);
            const SharedArray<float> device_b(queue(), b);
            SharedArray<float> device_c(queue(), c);
            const std::size_t scratch_bytes = std::size_t{16} << 20;
            SharedArray<std::byte> scratch(queue(), scratch_bytes);

            Kernels::SYCL::gemm<float>(queue(), Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                       Core::Transpose::NoTrans, m, n, k, 1.0f, device_a.get(),
                                       k, 0, device_b.get(), n, 0, 2.0f, device_c.get(), n, 0, 1,
                                       scratch.get(), scratch_bytes, {}).wait();
            Kernels::Reference::gemm<float>(Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                            Core::Transpose::NoTrans, m, n, k, 1.0f, a.data(), k,
                                            b.data(), n, 2.0f, c.data(), n);

            EXPECT_EQ(device_c.values(), c);
        }
    } // namespace Test
} // namespace SushiBLAS
