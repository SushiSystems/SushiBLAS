/**************************************************************************/
/* test_spmm.cpp                                                          */
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
 * @file test_spmm.cpp
 * @brief Tests the SYCL spmm against the reference spmm in both layouts and both forms.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/reference/spmm.hpp>
#include <SushiBLAS/sparse_kernels/sycl/spmm.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include <SushiBLAS/sparse_fixtures/integer_csr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs spmm on the host CPU queue. */
        class Medium_SYCLSpmm : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLSpmm, MatchesTheReferenceForEveryLayoutAndForm)
        {
            const int64_t rows = 97;
            const int64_t cols = 83;
            const int64_t nrhs = 5;
            const IntegerCsr a = integer_csr(rows, cols, 2, false);
            const SharedArray<int64_t> offsets(queue(), a.offsets);
            const SharedArray<int64_t> columns(queue(), a.columns);
            const SharedArray<double> values(queue(), a.values);
            const Kernels::CsrPattern<int64_t> device_pattern{rows, cols, a.nnz(), 0,
                                                              offsets.get(), columns.get()};
            const Kernels::CsrPattern<int64_t> host_pattern{rows, cols, a.nnz(), 0,
                                                            a.offsets.data(), a.columns.data()};
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                for (const bool transposed : {false, true})
                {
                    SCOPED_TRACE(transposed);
                    const int64_t in_rows = transposed ? rows : cols;
                    const int64_t out_rows = transposed ? cols : rows;
                    const bool row_major = (layout == Core::Layout::RowMajor);
                    const std::vector<double> b = small_integers<double>(in_rows * nrhs, 3);
                    std::vector<double> c = small_integers<double>(out_rows * nrhs, 4);
                    const SharedArray<double> device_b(queue(), b);
                    SharedArray<double> device_c(queue(), c);
                    SharedArray<Kernels::Accumulator<double>> work(queue(), c.size());

                    Kernels::SYCL::spmm(queue(), layout, transpose_of(transposed), device_pattern,
                                        values.get(), nrhs, 2.0, device_b.get(),
                                        row_major ? nrhs : in_rows, 1.0, device_c.get(),
                                        row_major ? nrhs : out_rows, work.get(), {}).wait();
                    Kernels::Reference::spmm(layout, transpose_of(transposed), host_pattern,
                                             a.values.data(), nrhs, 2.0, b.data(),
                                             row_major ? nrhs : in_rows, 1.0, c.data(),
                                             row_major ? nrhs : out_rows);

                    EXPECT_EQ(device_c.values(), c);
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
