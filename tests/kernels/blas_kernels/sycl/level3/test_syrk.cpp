/**************************************************************************/
/* test_syrk.cpp                                                          */
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
 * @file test_syrk.cpp
 * @brief Tests the tiled and the packed SYCL syrk against the reference syrk on the host device.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level3/syrk.hpp>
#include <SushiBLAS/blas_kernels/sycl/level3/syrk.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The order of C; odd, so no tile divides it.
        constexpr int64_t K_ORDER = 67;

        /// @brief The inner dimension of the update.
        constexpr int64_t K_DEPTH = 45;

        /// @brief The scratch the packed path may take; none sends the call down the tiled path.
        constexpr std::size_t K_SCRATCH_BYTES = std::size_t{8} << 20;

        /** @brief Runs syrk on the host CPU queue. */
        class Device_SYCLSyrk : public HostDeviceTest
        {

        };

        TEST_F(Device_SYCLSyrk, BothPathsMatchTheReferenceForEveryFlag)
        {
            // Every operand is a small integer, so both kernels compute without rounding.
            const std::vector<double> a = small_integers<double>(K_ORDER * K_DEPTH, 1);
            const SharedArray<double> device_a(queue(), a);
            SharedArray<std::byte> scratch(queue(), K_SCRATCH_BYTES);
            for (const std::size_t scratch_bytes : {std::size_t{0}, K_SCRATCH_BYTES})
            {
                for (const Core::Layout layout :
                     {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
                {
                    for (const bool upper : {false, true})
                    {
                        for (const bool transposed : {false, true})
                        {
                            SCOPED_TRACE(4 * (scratch_bytes > 0) + 2 * upper + transposed);
                            const bool by_rows = (layout == Core::Layout::RowMajor) != transposed;
                            const int64_t lda = by_rows ? K_DEPTH : K_ORDER;
                            std::vector<double> c = small_integers<double>(K_ORDER * K_ORDER, 2);
                            SharedArray<double> device_c(queue(), c);

                            Kernels::SYCL::syrk(queue(), layout, uplo_of(upper),
                                                transpose_of(transposed), K_ORDER, K_DEPTH, 2.0,
                                                device_a.get(), lda, 0, -1.0, device_c.get(),
                                                K_ORDER, 0, 1,
                                                scratch_bytes ? scratch.get() : nullptr,
                                                scratch_bytes, {}).wait();
                            Kernels::Reference::syrk(layout, uplo_of(upper),
                                                     transpose_of(transposed), K_ORDER, K_DEPTH,
                                                     2.0, a.data(), lda, -1.0, c.data(), K_ORDER);

                            EXPECT_EQ(device_c.values(), c);
                        }
                    }
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
