/**************************************************************************/
/* test_symv.cpp                                                          */
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
 * @file test_symv.cpp
 * @brief Tests the reference symv against a dense loop over the mirrored triangle.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/symv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_ReferenceSymv, ReadsOneTriangleAndMirrorsIt)
        {
            const int64_t n = 5;
            // The triangle not named by uplo holds 99, which the kernel must never read.
            const std::vector<double> a = small_integers<double>(n * n, 1);
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                for (const bool upper : {false, true})
                {
                    SCOPED_TRACE(upper);
                    std::vector<double> poisoned = a;
                    for (int64_t i = 0; i < n; ++i)
                        for (int64_t j = 0; j < n; ++j)
                            if (upper ? (i > j) : (i < j))
                                poisoned[static_cast<std::size_t>(i * n + j)] = 99.0;
                    const std::vector<double> x = small_integers<double>(n, 2);
                    std::vector<double> y = small_integers<double>(n, 3);
                    const std::vector<double> expected =
                        dense_matvec(symmetric_from(a, n, upper), n, n, false, 2.0, x, 3.0, y);

                    Kernels::Reference::symv(layout, uplo_of(upper), n, 2.0,
                                             in_layout(poisoned, n, n, layout).data(), n,
                                             x.data(), 1, 3.0, y.data(), 1);

                    EXPECT_EQ(y, expected);
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
