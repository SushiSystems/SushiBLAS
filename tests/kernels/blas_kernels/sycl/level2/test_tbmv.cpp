/**************************************************************************/
/* test_tbmv.cpp                                                          */
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
 * @file test_tbmv.cpp
 * @brief Tests the SYCL tbmv against the reference tbmv for every flag on the host device.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/tbmv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/tbmv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs tbmv on the host CPU queue. */
        class Medium_SYCLTbmv : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLTbmv, MatchesTheReferenceForEveryFlag)
        {
            const int64_t n = 44;
            const int64_t k = 3;
            const int64_t ld = k + 1;
            const std::vector<double> band = small_integers<double>(n * ld, 1);
            const SharedArray<double> device_band(queue(), band);
            SharedArray<double> work(queue(), static_cast<std::size_t>(n));
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                for (const bool upper : {false, true})
                {
                    for (const bool transposed : {false, true})
                    {
                        for (const bool unit : {false, true})
                        {
                            SCOPED_TRACE(4 * upper + 2 * transposed + unit);
                            std::vector<double> x = small_integers<double>(n, 2);
                            SharedArray<double> device_x(queue(), x);

                            Kernels::SYCL::tbmv(queue(), layout, uplo_of(upper),
                                                transpose_of(transposed), diag_of(unit), n, k,
                                                device_band.get(), ld, device_x.get(), 1,
                                                work.get(), {}).wait();
                            Kernels::Reference::tbmv(layout, uplo_of(upper),
                                                     transpose_of(transposed), diag_of(unit), n,
                                                     k, band.data(), ld, x.data(), 1);

                            EXPECT_EQ(device_x.values(), x);
                        }
                    }
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
