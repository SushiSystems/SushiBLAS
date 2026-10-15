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
 * @brief Tests the reference spsymv against a dense product with the mirrored triangle.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/reference/spsymv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include <SushiBLAS/sparse_fixtures/integer_csr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_ReferenceSpsymv, ReadsOneTriangleAndSkipsTheOther)
        {
            const int64_t n = 8;
            const IntegerCsr a = integer_csr(n, n, 4, true);
            const Kernels::CsrPattern<int64_t> pattern{n, n, a.nnz(), 0, a.offsets.data(),
                                                       a.columns.data()};
            const std::vector<double> x = small_integers<double>(n, 5);
            for (const bool upper : {false, true})
            {
                SCOPED_TRACE(upper);
                std::vector<double> y = small_integers<double>(n, 6);
                const std::vector<double> expected =
                    dense_matvec(symmetric_from(a.dense(), n, upper), n, n, false, 1.0, x, 2.0, y);

                Kernels::Reference::spsymv(uplo_of(upper), pattern, a.values.data(), 1.0,
                                           x.data(), 1, 2.0, y.data(), 1);

                EXPECT_EQ(y, expected);
            }
        }
    } // namespace Test
} // namespace SushiBLAS
