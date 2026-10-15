/**************************************************************************/
/* test_sbmv.cpp                                                          */
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
 * @file test_sbmv.cpp
 * @brief Tests the reference sbmv against a dense loop over the mirrored band triangle.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/sbmv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_ReferenceSbmv, MatchesADenseLoopOverTheBandTriangle)
        {
            const int64_t n = 6;
            const int64_t k = 2;
            const int64_t ld = k + 2;
            const std::vector<double> a = small_integers<double>(n * n, 1);
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                for (const bool upper : {false, true})
                {
                    SCOPED_TRACE(upper);
                    const std::vector<double> symmetric =
                        band_of(symmetric_from(a, n, upper), n, n, k, k);
                    const std::vector<double> x = small_integers<double>(n, 2);
                    std::vector<double> y = small_integers<double>(n, 3);
                    const std::vector<double> expected =
                        dense_matvec(symmetric, n, n, false, -1.0, x, 2.0, y);
                    const std::vector<double> band =
                        to_triangle_band(symmetric, n, uplo_of(upper), k, ld, layout, 99.0);

                    Kernels::Reference::sbmv(layout, uplo_of(upper), n, k, -1.0, band.data(), ld,
                                             x.data(), 1, 2.0, y.data(), 1);

                    EXPECT_EQ(y, expected);
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
