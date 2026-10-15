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
 * @file test_packed_product.cpp
 * @brief Tests that the packed product hands every sum of C, and only those, to its store.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/packed_product_plan.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_seam_kernels/reference/level3/gemm.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/packed_product.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/packed_product_harness/packed_product_harness.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Writes each sum into a row-major C of a given width and counts the calls. */
        struct CountingStore
        {
            int64_t width = 0;        /**< Columns of C. */
            unsigned* calls = nullptr; /**< One counter per element of C. */

            /** @brief Accepts every block. */
            [[nodiscard]] bool wants(int64_t, int64_t, int64_t, int64_t) const noexcept
            {
                return true;
            }

            /** @brief Stores sum at (row, col) and counts the store. */
            void operator()(double* c, int64_t row, int64_t col,
                            Kernels::Accumulator<double> sum) const
            {
                c[row * width + col] = sum;
                calls[row * width + col] += 1;
            }
        };

        /** @brief Runs the packed product on the host CPU queue. */
        class Medium_SYCLPackedProduct : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLPackedProduct, StoresEverySumOfCOnce)
        {
            const int64_t m = 29;
            const int64_t n = 31;
            const int64_t k = 17;
            const std::vector<double> a = small_integers<double>(m * k, 1);
            const std::vector<double> b = small_integers<double>(k * n, 2);
            const SharedArray<double> device_a(queue(), a);
            const SharedArray<double> device_b(queue(), b);
            SharedArray<double> device_c(queue(), static_cast<std::size_t>(m * n));
            SharedArray<unsigned> calls(queue(), static_cast<std::size_t>(m * n));
            const Kernels::PackedProductPlan plan =
                Kernels::plan_packed_blocks<double>(ragged_blocking<double>(), 4, m, n, k, 1);
            ASSERT_GT(plan.blocks, 0);
            SharedArray<std::byte> scratch(queue(), plan.scratch_bytes);
            const Kernels::OperandAccess a_access =
                Kernels::operand_access(Core::Layout::RowMajor, Core::Transpose::NoTrans, k);
            const Kernels::OperandAccess b_access =
                Kernels::operand_access(Core::Layout::RowMajor, Core::Transpose::NoTrans, n);

            Kernels::SYCL::Detail::packed_product<double>(
                queue(), plan, m, n, k, device_a.get(), a_access, 0, device_b.get(), b_access, 0,
                device_c.get(), 0, scratch.get(), {}, CountingStore{n, calls.get()}).wait();
            std::vector<double> expected(static_cast<std::size_t>(m * n), 0.0);
            Kernels::Reference::gemm<double>(Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                             Core::Transpose::NoTrans, m, n, k, 1.0, a.data(), k,
                                             b.data(), n, 0.0, expected.data(), n);

            EXPECT_EQ(device_c.values(), expected);
            EXPECT_EQ(calls.values(), std::vector<unsigned>(static_cast<std::size_t>(m * n), 1u));
        }
    } // namespace Test
} // namespace SushiBLAS
