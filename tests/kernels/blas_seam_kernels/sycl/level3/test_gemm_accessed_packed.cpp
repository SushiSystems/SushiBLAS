/**************************************************************************/
/* test_gemm_accessed_packed.cpp                                          */
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
 * @file test_gemm_accessed_packed.cpp
 * @brief Tests the accessed SYCL gemm, with and without scratch, against the reference gemm.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_seam_kernels/reference/level3/gemm.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm_accessed_packed.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the accessed gemm on the host CPU queue. */
        class Device_SYCLGemmAccessedPacked : public HostDeviceTest
        {

        };

        TEST_F(Device_SYCLGemmAccessedPacked, BothPathsMatchTheReference)
        {
            // Every operand is a small integer, so both kernels compute without rounding.
            const int64_t m = 61;
            const int64_t n = 47;
            const int64_t k = 39;
            const std::vector<double> a = small_integers<double>(k * m, 1);
            const std::vector<double> b = small_integers<double>(k * n, 2);
            const SharedArray<double> device_a(queue(), a);
            const SharedArray<double> device_b(queue(), b);
            // A is stored k x m and read transposed; B is k x n, read as stored.
            const Kernels::OperandAccess a_access =
                Kernels::operand_access(Core::Layout::RowMajor, Core::Transpose::Trans, m);
            const Kernels::OperandAccess b_access =
                Kernels::operand_access(Core::Layout::RowMajor, Core::Transpose::NoTrans, n);
            const Kernels::OperandAccess c_access =
                Kernels::operand_access(Core::Layout::RowMajor, Core::Transpose::NoTrans, n);
            const std::size_t scratch_bytes = std::size_t{16} << 20;
            SharedArray<std::byte> scratch(queue(), scratch_bytes);
            for (const std::size_t bytes : {std::size_t{0}, scratch_bytes})
            {
                SCOPED_TRACE(bytes);
                std::vector<double> c = small_integers<double>(m * n, 3);
                SharedArray<double> device_c(queue(), c);

                Kernels::SYCL::gemm_accessed<double>(
                    queue(), m, n, k, 2.0, device_a.get(), a_access, 0, device_b.get(), b_access,
                    0, -1.0, device_c.get(), c_access, 0, 1, bytes ? scratch.get() : nullptr,
                    bytes, {}).wait();
                Kernels::Reference::gemm_accessed<double>(m, n, k, 2.0, a.data(), a_access,
                                                          b.data(), b_access, -1.0, c.data(),
                                                          c_access);

                EXPECT_EQ(device_c.values(), c);
            }
        }
    } // namespace Test
} // namespace SushiBLAS
