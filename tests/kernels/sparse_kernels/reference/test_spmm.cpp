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
 * @brief Tests the reference spmm column by column against the reference spmv, in both layouts.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/reference/spmm.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include <SushiBLAS/sparse_fixtures/integer_csr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_ReferenceSpmm, EachColumnOfCIsAProductOfTheMatrixWithAColumnOfB)
        {
            const int64_t rows = 7;
            const int64_t cols = 6;
            const int64_t nrhs = 3;
            const IntegerCsr a = integer_csr(rows, cols, 2, false);
            const Kernels::CsrPattern<int64_t> pattern{rows, cols, a.nnz(), 0, a.offsets.data(),
                                                       a.columns.data()};
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                for (const bool transposed : {false, true})
                {
                    SCOPED_TRACE(transposed);
                    const int64_t in_rows = transposed ? rows : cols;
                    const int64_t out_rows = transposed ? cols : rows;
                    const std::vector<double> b = small_integers<double>(in_rows * nrhs, 3);
                    const std::vector<double> c = small_integers<double>(out_rows * nrhs, 4);
                    std::vector<double> expected(static_cast<std::size_t>(out_rows * nrhs));
                    for (int64_t j = 0; j < nrhs; ++j)
                    {
                        std::vector<double> b_column(static_cast<std::size_t>(in_rows));
                        std::vector<double> c_column(static_cast<std::size_t>(out_rows));
                        for (int64_t i = 0; i < in_rows; ++i)
                            b_column[static_cast<std::size_t>(i)] = element(b, nrhs, i, j);
                        for (int64_t i = 0; i < out_rows; ++i)
                            c_column[static_cast<std::size_t>(i)] = element(c, nrhs, i, j);
                        const std::vector<double> product = dense_matvec(
                            a.dense(), rows, cols, transposed, 2.0, b_column, 1.0, c_column);
                        for (int64_t i = 0; i < out_rows; ++i)
                            expected[static_cast<std::size_t>(i * nrhs + j)] =
                                product[static_cast<std::size_t>(i)];
                    }
                    const bool row_major = (layout == Core::Layout::RowMajor);
                    const std::vector<double> stored_b = in_layout(b, in_rows, nrhs, layout);
                    std::vector<double> stored_c = in_layout(c, out_rows, nrhs, layout);

                    Kernels::Reference::spmm(layout, transpose_of(transposed), pattern,
                                             a.values.data(), nrhs, 2.0, stored_b.data(),
                                             row_major ? nrhs : in_rows, 1.0, stored_c.data(),
                                             row_major ? nrhs : out_rows);

                    EXPECT_EQ(stored_c, in_layout(expected, out_rows, nrhs, layout));
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
