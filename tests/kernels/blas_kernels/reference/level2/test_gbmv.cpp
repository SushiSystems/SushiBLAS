/**************************************************************************/
/* test_gbmv.cpp                                                          */
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
 * @file test_gbmv.cpp
 * @brief Tests the reference gbmv against a loop over the band, in both layouts and transposes.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/gbmv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_ReferenceGbmv, MatchesADenseLoopOverTheBand)
        {
            const int64_t m = 5;
            const int64_t n = 6;
            const int64_t kl = 1;
            const int64_t ku = 2;
            const int64_t ld = kl + ku + 2;
            const std::vector<double> dense = band_of(small_integers<double>(m * n, 1), m, n,
                                                      kl, ku);
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                for (const bool transposed : {false, true})
                {
                    SCOPED_TRACE(transposed);
                    const std::vector<double> x =
                        small_integers<double>(transposed ? m : n, 2);
                    std::vector<double> y = small_integers<double>(transposed ? n : m, 3);
                    const std::vector<double> expected =
                        dense_matvec(dense, m, n, transposed, 2.0, x, 1.0, y);
                    const std::vector<double> band =
                        to_band(dense, m, n, kl, ku, ld, layout, 99.0);

                    Kernels::Reference::gbmv(layout, transpose_of(transposed), m, n, kl, ku, 2.0,
                                             band.data(), ld, x.data(), 1, 1.0, y.data(), 1);

                    EXPECT_EQ(y, expected);
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
