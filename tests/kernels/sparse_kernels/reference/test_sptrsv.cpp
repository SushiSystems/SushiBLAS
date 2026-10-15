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
 * @brief Tests that the reference sptrsv recovers x from op(A) * x for either triangle.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/reference/sptrsv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include <SushiBLAS/sparse_fixtures/integer_csr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_ReferenceSptrsv, RecoversTheVectorAProductMade)
        {
            const int64_t n = 9;
            // A stored unit diagonal keeps every substitution exact in integers.
            const IntegerCsr a = integer_csr(n, n, 3, true);
            const Kernels::CsrPattern<int64_t> pattern{n, n, a.nnz(), 0, a.offsets.data(),
                                                       a.columns.data()};
            const std::vector<double> solution = small_integers<double>(n, 4);
            for (const bool upper : {false, true})
            {
                for (const bool transposed : {false, true})
                {
                    for (const bool unit : {false, true})
                    {
                        SCOPED_TRACE(4 * upper + 2 * transposed + unit);
                        std::vector<double> b = dense_matvec(triangle_of(a.dense(), n, upper, unit),
                                                             n, n, transposed, 1.0, solution, 0.0,
                                                             solution);

                        Kernels::Reference::sptrsv(uplo_of(upper), transpose_of(transposed),
                                                   diag_of(unit), pattern, a.values.data(),
                                                   b.data());

                        EXPECT_EQ(b, solution);
                    }
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
