/**************************************************************************/
/* test_trmv.cpp                                                          */
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
 * @file test_trmv.cpp
 * @brief Tests the SYCL trmv against the reference trmv for every flag on the host device.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/trmv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/trmv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs trmv on the host CPU queue. */
        class Medium_SYCLTrmv : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLTrmv, MatchesTheReferenceForEveryFlag)
        {
            const int64_t n = 47;
            const std::vector<double> a = small_integers<double>(n * n, 1);
            const SharedArray<double> device_a(queue(), a);
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
                            std::vector<double> x = small_integers<double>(2 * n, 2);
                            SharedArray<double> device_x(queue(), x);

                            Kernels::SYCL::trmv(queue(), layout, uplo_of(upper),
                                                transpose_of(transposed), diag_of(unit), n,
                                                device_a.get(), n, device_x.get(), 2, work.get(),
                                                {}).wait();
                            Kernels::Reference::trmv(layout, uplo_of(upper),
                                                     transpose_of(transposed), diag_of(unit), n,
                                                     a.data(), n, x.data(), 2);

                            EXPECT_EQ(device_x.values(), x);
                        }
                    }
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
