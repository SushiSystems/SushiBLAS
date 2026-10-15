/**************************************************************************/
/* test_triangular_product.cpp                                            */
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
 * @file test_triangular_product.cpp
 * @brief Tests the SYCL triangular product against the reference one through a packed access.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_seam_kernels/packed_access.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/triangular_product.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/triangular_product.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the triangular product on the host CPU queue. */
        class Device_SYCLTriangularProduct : public HostDeviceTest
        {

        };

        TEST_F(Device_SYCLTriangularProduct, MatchesTheReferenceForEveryFlag)
        {
            const int64_t n = 33;
            const std::vector<double> packed = small_integers<double>(n * (n + 1) / 2, 1);
            const SharedArray<double> device_packed(queue(), packed);
            SharedArray<double> work(queue(), static_cast<std::size_t>(n));
            for (const bool upper : {false, true})
            {
                for (const bool transposed : {false, true})
                {
                    for (const bool unit : {false, true})
                    {
                        SCOPED_TRACE(4 * upper + 2 * transposed + unit);
                        const Kernels::PackedAccess access = Kernels::packed_access(
                            Core::Layout::ColumnMajor, uplo_of(upper), transpose_of(transposed),
                            n);
                        std::vector<double> x = small_integers<double>(n, 2);
                        SharedArray<double> device_x(queue(), x);

                        Kernels::SYCL::Detail::triangular_product(
                            queue(), access, unit, n, device_packed.get(), device_x.get(), 1,
                            work.get(), {}).wait();
                        Kernels::Reference::Detail::triangular_product(
                            access, transposed != upper, unit, n, packed.data(), x.data(), 1);

                        EXPECT_EQ(device_x.values(), x);
                    }
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
