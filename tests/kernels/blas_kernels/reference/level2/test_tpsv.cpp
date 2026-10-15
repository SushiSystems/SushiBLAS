/**************************************************************************/
/* test_tpsv.cpp                                                          */
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
 * @file test_tpsv.cpp
 * @brief Tests that the reference tpsv recovers x from op(A) * x for a packed triangle.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/tpsv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_ReferenceTpsv, RecoversTheVectorAProductMade)
        {
            const int64_t n = 5;
            // A unit diagonal makes every substitution exact in integers.
            const std::vector<double> a = unit_diagonal_matrix<double>(n, 1);
            const std::vector<double> solution = small_integers<double>(n, 2);
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                for (const bool upper : {false, true})
                {
                    for (const bool transposed : {false, true})
                    {
                        for (const bool unit : {false, true})
                        {
                            SCOPED_TRACE(4 * upper + 2 * transposed + unit);
                            std::vector<double> x =
                                dense_matvec(triangle_of(a, n, upper, false), n, n, transposed,
                                             1.0, solution, 0.0, solution);
                            const std::vector<double> packed =
                                to_packed(a, n, uplo_of(upper), layout);

                            Kernels::Reference::tpsv(layout, uplo_of(upper),
                                                     transpose_of(transposed), diag_of(unit), n,
                                                     packed.data(), x.data(), 1);

                            EXPECT_EQ(x, solution);
                        }
                    }
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
