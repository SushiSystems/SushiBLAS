/**************************************************************************/
/* test_syr2k.cpp                                                         */
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
 * @file test_syr2k.cpp
 * @brief Tests the SYCL syr2k against the reference syr2k on the host device, batched included.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level3/syr2k.hpp>
#include <SushiBLAS/blas_kernels/sycl/level3/syr2k.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The order of C; odd, so no tile divides it.
        constexpr int64_t K_ORDER = 53;

        /// @brief The inner dimension of the update.
        constexpr int64_t K_DEPTH = 29;

        /// @brief The matrices of the batch.
        constexpr int64_t K_BATCH = 2;

        /** @brief Runs syr2k on the host CPU queue. */
        class Medium_SYCLSyr2k : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLSyr2k, MatchesTheReferenceForEveryFlag)
        {
            const int64_t panel = K_ORDER * K_DEPTH;
            const std::vector<double> a = small_integers<double>(K_BATCH * panel, 1);
            const std::vector<double> b = small_integers<double>(K_BATCH * panel, 2);
            const SharedArray<double> device_a(queue(), a);
            const SharedArray<double> device_b(queue(), b);
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                for (const bool upper : {false, true})
                {
                    for (const bool transposed : {false, true})
                    {
                        SCOPED_TRACE(2 * upper + transposed);
                        const bool by_rows = (layout == Core::Layout::RowMajor) != transposed;
                        const int64_t ld = by_rows ? K_DEPTH : K_ORDER;
                        const int64_t square = K_ORDER * K_ORDER;
                        std::vector<double> c = small_integers<double>(K_BATCH * square, 3);
                        SharedArray<double> device_c(queue(), c);

                        Kernels::SYCL::syr2k(queue(), layout, uplo_of(upper),
                                             transpose_of(transposed), K_ORDER, K_DEPTH, 3.0,
                                             device_a.get(), ld, panel, device_b.get(), ld,
                                             panel, 2.0, device_c.get(), K_ORDER, square,
                                             K_BATCH, {}).wait();
                        Kernels::Reference::syr2k(layout, uplo_of(upper), transpose_of(transposed),
                                                  K_ORDER, K_DEPTH, 3.0, a.data(), ld, panel,
                                                  b.data(), ld, panel, 2.0, c.data(), K_ORDER,
                                                  square, K_BATCH);

                        EXPECT_EQ(device_c.values(), c);
                    }
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
