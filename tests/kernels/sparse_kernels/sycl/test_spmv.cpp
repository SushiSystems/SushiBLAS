/**************************************************************************/
/* test_spmv.cpp                                                          */
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
 * @file test_spmv.cpp
 * @brief Tests the SYCL spmv, plain and scattered transposed, against the reference spmv.
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
#include <SushiBLAS/sparse_kernels/reference/spmv.hpp>
#include <SushiBLAS/sparse_kernels/sycl/row_blocks.hpp>
#include <SushiBLAS/sparse_kernels/sycl/spmv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include <SushiBLAS/sparse_fixtures/integer_csr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs spmv on the host CPU queue. */
        class Device_SYCLSpmv : public HostDeviceTest
        {

        };

        TEST_F(Device_SYCLSpmv, MatchesTheReferencePlainAndTransposed)
        {
            // Every value is a small integer, so both kernels compute without rounding.
            const int64_t rows = 311;
            const int64_t cols = 257;
            const IntegerCsr a = integer_csr(rows, cols, 1, false);
            const SharedArray<int64_t> offsets(queue(), a.offsets);
            const SharedArray<int64_t> columns(queue(), a.columns);
            const SharedArray<double> values(queue(), a.values);
            const Kernels::CsrPattern<int64_t> device_pattern{rows, cols, a.nnz(), 0,
                                                              offsets.get(), columns.get()};
            const Kernels::CsrPattern<int64_t> host_pattern{rows, cols, a.nnz(), 0,
                                                            a.offsets.data(), a.columns.data()};
            for (const bool transposed : {false, true})
            {
                SCOPED_TRACE(transposed);
                const int64_t outputs = transposed ? cols : rows;
                const std::size_t blocks = Kernels::SYCL::scatter_blocks(
                    Device::profile_for(queue()), outputs, a.nnz());
                SharedArray<Kernels::Accumulator<double>> partials(
                    queue(), blocks * static_cast<std::size_t>(outputs));
                const std::vector<double> x = small_integers<double>(transposed ? rows : cols, 2);
                std::vector<double> y = small_integers<double>(outputs, 3);
                const SharedArray<double> device_x(queue(), x);
                SharedArray<double> device_y(queue(), y);

                Kernels::SYCL::spmv(queue(), transpose_of(transposed), device_pattern,
                                    values.get(), 2.0, device_x.get(), 1, -1.0, device_y.get(), 1,
                                    partials.get(), blocks, {}).wait();
                Kernels::Reference::spmv(transpose_of(transposed), host_pattern, a.values.data(),
                                         2.0, x.data(), 1, -1.0, y.data(), 1);

                EXPECT_EQ(device_y.values(), y);
            }
        }
    } // namespace Test
} // namespace SushiBLAS
