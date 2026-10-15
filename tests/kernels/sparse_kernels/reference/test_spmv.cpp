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
 * @brief Tests the reference spmv against a dense product, plain and transposed.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/reference/spmv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include <SushiBLAS/sparse_fixtures/integer_csr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_ReferenceSpmv, MatchesADenseProductPlainAndTransposed)
        {
            const IntegerCsr a = integer_csr(9, 7, 1, false);
            const Kernels::CsrPattern<int64_t> pattern{a.rows, a.cols, a.nnz(), 0,
                                                       a.offsets.data(), a.columns.data()};
            for (const bool transposed : {false, true})
            {
                SCOPED_TRACE(transposed);
                const std::vector<double> x = small_integers<double>(transposed ? 9 : 7, 2);
                std::vector<double> y = small_integers<double>(transposed ? 7 : 9, 3);
                const std::vector<double> expected =
                    dense_matvec(a.dense(), 9, 7, transposed, 2.0, x, -1.0, y);

                Kernels::Reference::spmv(transpose_of(transposed), pattern, a.values.data(), 2.0,
                                         x.data(), 1, -1.0, y.data(), 1);

                EXPECT_EQ(y, expected);
            }
        }
    } // namespace Test
} // namespace SushiBLAS
