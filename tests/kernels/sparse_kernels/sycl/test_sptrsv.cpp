/**************************************************************************/
/* test_sptrsv.cpp                                                        */
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
 * @file test_sptrsv.cpp
 * @brief Tests the SYCL sptrsv against the reference sptrsv for every flag on the host device.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/reference/sptrsv.hpp>
#include <SushiBLAS/sparse_kernels/sycl/sptrsv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include <SushiBLAS/sparse_fixtures/integer_csr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs sptrsv on the host CPU queue. */
        class Device_SYCLSptrsv : public HostDeviceTest
        {

        };

        TEST_F(Device_SYCLSptrsv, MatchesTheReferenceForEveryFlag)
        {
            const int64_t n = 64;
            // A stored unit diagonal keeps every substitution exact in integers.
            const IntegerCsr a = integer_csr(n, n, 7, true);
            const SharedArray<int64_t> offsets(queue(), a.offsets);
            const SharedArray<int64_t> columns(queue(), a.columns);
            const SharedArray<double> values(queue(), a.values);
            const Kernels::CsrPattern<int64_t> device_pattern{n, n, a.nnz(), 0, offsets.get(),
                                                              columns.get()};
            const Kernels::CsrPattern<int64_t> host_pattern{n, n, a.nnz(), 0, a.offsets.data(),
                                                            a.columns.data()};
            SharedArray<Kernels::Accumulator<double>> work(queue(), static_cast<std::size_t>(n));
            const std::vector<double> solution = small_integers<double>(n, 8);
            for (const bool upper : {false, true})
            {
                for (const bool transposed : {false, true})
                {
                    for (const bool unit : {false, true})
                    {
                        SCOPED_TRACE(4 * upper + 2 * transposed + unit);
                        std::vector<double> b =
                            dense_matvec(triangle_of(a.dense(), n, upper, unit), n, n, transposed,
                                         1.0, solution, 0.0, solution);
                        SharedArray<double> device_b(queue(), b);

                        Kernels::SYCL::sptrsv(queue(), uplo_of(upper), transpose_of(transposed),
                                              diag_of(unit), device_pattern, values.get(),
                                              device_b.get(), work.get(), {}).wait();
                        Kernels::Reference::sptrsv(uplo_of(upper), transpose_of(transposed),
                                                   diag_of(unit), host_pattern, a.values.data(),
                                                   b.data());

                        EXPECT_EQ(device_b.values(), b);
                        EXPECT_EQ(b, solution);
                    }
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
