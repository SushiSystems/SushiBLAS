/**************************************************************************/
/* test_symv.cpp                                                          */
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
 * @file test_symv.cpp
 * @brief Tests the SYCL symv against the reference symv on the host device.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/symv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/symv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs symv on the host CPU queue. */
        class Medium_SYCLSymv : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLSymv, MatchesTheReferenceForBothTrianglesAndLayouts)
        {
            const int64_t n = 45;
            const std::vector<double> a = small_integers<double>(n * n, 1);
            const std::vector<double> x = small_integers<double>(n, 2);
            const SharedArray<double> device_a(queue(), a);
            const SharedArray<double> device_x(queue(), x);
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                for (const bool upper : {false, true})
                {
                    SCOPED_TRACE(upper);
                    std::vector<double> y = small_integers<double>(n, 3);
                    SharedArray<double> device_y(queue(), y);

                    Kernels::SYCL::symv(queue(), layout, uplo_of(upper), n, 2.0, device_a.get(),
                                        n, device_x.get(), 1, 3.0, device_y.get(), 1, {}).wait();
                    Kernels::Reference::symv(layout, uplo_of(upper), n, 2.0, a.data(), n,
                                             x.data(), 1, 3.0, y.data(), 1);

                    EXPECT_EQ(device_y.values(), y);
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
