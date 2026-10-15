/**************************************************************************/
/* test_symmetric_product.cpp                                             */
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
 * @file test_symmetric_product.cpp
 * @brief Tests the SYCL symmetric product against the reference one through a packed access.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/blas_seam_kernels/packed_access.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/symmetric_product.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/symmetric_product.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the symmetric product on the host CPU queue. */
        class Medium_SYCLSymmetricProduct : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLSymmetricProduct, MatchesTheReferenceForBothTriangles)
        {
            const int64_t n = 35;
            const std::vector<double> packed = small_integers<double>(n * (n + 1) / 2, 1);
            const std::vector<double> x = small_integers<double>(n, 2);
            const SharedArray<double> device_packed(queue(), packed);
            const SharedArray<double> device_x(queue(), x);
            for (const bool upper : {false, true})
            {
                SCOPED_TRACE(upper);
                const Kernels::PackedAccess access = Kernels::packed_access(
                    Core::Layout::RowMajor, uplo_of(upper), Core::Transpose::NoTrans, n);
                std::vector<double> y = small_integers<double>(n, 3);
                SharedArray<double> device_y(queue(), y);

                Kernels::SYCL::Detail::symmetric_product<double, Kernels::Unconjugated>(
                    queue(), access, n, n, 2.0, device_packed.get(), device_x.get(), 1, -1.0,
                    device_y.get(), 1, {}).wait();
                Kernels::Reference::Detail::symmetric_product<double, Kernels::Unconjugated>(
                    access, n, 2.0, packed.data(), x.data(), 1, -1.0, y.data(), 1);

                EXPECT_EQ(device_y.values(), y);
            }
        }
    } // namespace Test
} // namespace SushiBLAS
