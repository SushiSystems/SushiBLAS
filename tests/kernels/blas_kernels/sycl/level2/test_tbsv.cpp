/**************************************************************************/
/* test_tbsv.cpp                                                          */
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
 * @file test_tbsv.cpp
 * @brief Tests the SYCL tbsv against the reference tbsv for every flag on the host device.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/tbsv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/tbsv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs tbsv on the host CPU queue. */
        class Device_SYCLTbsv : public HostDeviceTest
        {

        };

        TEST_F(Device_SYCLTbsv, MatchesTheReferenceForEveryFlag)
        {
            const int64_t n = 42;
            const int64_t k = 2;
            const int64_t ld = k + 1;
            // A unit diagonal keeps every substitution exact in integers.
            const std::vector<double> a = band_of(unit_diagonal_matrix<double>(n, 1), n, n, k, k);
            SharedArray<Kernels::Accumulator<double>> work(queue(), static_cast<std::size_t>(n));
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                for (const bool upper : {false, true})
                {
                    const std::vector<double> band =
                        to_triangle_band(a, n, uplo_of(upper), k, ld, layout, 0.0);
                    const SharedArray<double> device_band(queue(), band);
                    for (const bool transposed : {false, true})
                    {
                        for (const bool unit : {false, true})
                        {
                            SCOPED_TRACE(4 * upper + 2 * transposed + unit);
                            // A small solution keeps every substitution step exact.
                            const std::vector<double> solution = small_integers<double>(n, 2);
                            std::vector<double> x =
                                dense_matvec(triangle_of(a, n, upper, false), n, n, transposed,
                                             1.0, solution, 0.0, solution);
                            SharedArray<double> device_x(queue(), x);

                            Kernels::SYCL::tbsv(queue(), layout, uplo_of(upper),
                                                transpose_of(transposed), diag_of(unit), n, k,
                                                device_band.get(), ld, device_x.get(), 1,
                                                work.get(), {}).wait();
                            Kernels::Reference::tbsv(layout, uplo_of(upper),
                                                     transpose_of(transposed), diag_of(unit), n,
                                                     k, band.data(), ld, x.data(), 1);

                            EXPECT_EQ(device_x.values(), x);
                            EXPECT_EQ(x, solution);
                        }
                    }
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
