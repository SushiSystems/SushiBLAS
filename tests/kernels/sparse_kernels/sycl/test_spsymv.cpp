/**************************************************************************/
/* test_spsymv.cpp                                                        */
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
 * @file test_spsymv.cpp
 * @brief Tests the scattered SYCL spsymv against the reference spsymv on the host device.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/reference/spsymv.hpp>
#include <SushiBLAS/sparse_kernels/sycl/row_blocks.hpp>
#include <SushiBLAS/sparse_kernels/sycl/spsymv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include <SushiBLAS/sparse_fixtures/integer_csr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs spsymv on the host CPU queue. */
        class Device_SYCLSpsymv : public HostDeviceTest
        {

        };

        TEST_F(Device_SYCLSpsymv, MatchesTheReferenceForEitherTriangle)
        {
            const int64_t n = 283;
            const IntegerCsr a = integer_csr(n, n, 4, true);
            const SharedArray<int64_t> offsets(queue(), a.offsets);
            const SharedArray<int64_t> columns(queue(), a.columns);
            const SharedArray<double> values(queue(), a.values);
            const Kernels::CsrPattern<int64_t> device_pattern{n, n, a.nnz(), 0, offsets.get(),
                                                              columns.get()};
            const Kernels::CsrPattern<int64_t> host_pattern{n, n, a.nnz(), 0, a.offsets.data(),
                                                            a.columns.data()};
            const std::size_t blocks =
                Kernels::SYCL::scatter_blocks(Device::profile_for(queue()), n, a.nnz());
            SharedArray<Kernels::Accumulator<double>> partials(
                queue(), blocks * static_cast<std::size_t>(n));
            const std::vector<double> x = small_integers<double>(n, 5);
            const SharedArray<double> device_x(queue(), x);
            for (const bool upper : {false, true})
            {
                SCOPED_TRACE(upper);
                std::vector<double> y = small_integers<double>(n, 6);
                SharedArray<double> device_y(queue(), y);

                Kernels::SYCL::spsymv(queue(), uplo_of(upper), device_pattern, values.get(), 1.0,
                                      device_x.get(), 1, 3.0, device_y.get(), 1, partials.get(),
                                      blocks, {}).wait();
                Kernels::Reference::spsymv(uplo_of(upper), host_pattern, a.values.data(), 1.0,
                                           x.data(), 1, 3.0, y.data(), 1);

                EXPECT_EQ(device_y.values(), y);
            }
        }
    } // namespace Test
} // namespace SushiBLAS
