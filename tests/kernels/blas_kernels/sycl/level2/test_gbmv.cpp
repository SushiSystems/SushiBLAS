/**************************************************************************/
/* test_gbmv.cpp                                                          */
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
 * @file test_gbmv.cpp
 * @brief Tests the SYCL gbmv against the reference gbmv on the host device.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/gbmv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/gbmv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs gbmv on the host CPU queue. */
        class Medium_SYCLGbmv : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLGbmv, MatchesTheReferenceInEveryLayoutAndTranspose)
        {
            // Every operand is a small integer, so both kernels compute without rounding.
            const int64_t m = 41;
            const int64_t n = 29;
            const int64_t kl = 3;
            const int64_t ku = 5;
            const int64_t ld = kl + ku + 1;
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                const int64_t lines = (layout == Core::Layout::ColumnMajor) ? n : m;
                const std::vector<double> band = small_integers<double>(lines * ld, 1);
                const SharedArray<double> device_band(queue(), band);
                for (const bool transposed : {false, true})
                {
                    SCOPED_TRACE(transposed);
                    const std::vector<double> x = small_integers<double>(transposed ? m : n, 2);
                    std::vector<double> y = small_integers<double>(transposed ? n : m, 3);
                    const SharedArray<double> device_x(queue(), x);
                    SharedArray<double> device_y(queue(), y);

                    Kernels::SYCL::gbmv(queue(), layout, transpose_of(transposed), m, n, kl, ku,
                                        2.0, device_band.get(), ld, device_x.get(), 1, -1.0,
                                        device_y.get(), 1, {}).wait();
                    Kernels::Reference::gbmv(layout, transpose_of(transposed), m, n, kl, ku, 2.0,
                                             band.data(), ld, x.data(), 1, -1.0, y.data(), 1);

                    EXPECT_EQ(device_y.values(), y);
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
