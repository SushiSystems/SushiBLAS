/**************************************************************************/
/* test_ger.cpp                                                           */
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
 * @file test_ger.cpp
 * @brief Tests the reference ger, A += alpha * x * y^T, in both layouts and with a conjugated y.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/ger.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_ReferenceGer, AddsTheOuterProductInEitherLayout)
        {
            const int64_t m = 3;
            const int64_t n = 4;
            const std::vector<double> a = small_integers<double>(m * n, 1);
            const std::vector<double> x = small_integers<double>(m, 2);
            const std::vector<double> y = small_integers<double>(n, 3);
            std::vector<double> expected = a;
            for (int64_t i = 0; i < m; ++i)
                for (int64_t j = 0; j < n; ++j)
                    expected[static_cast<std::size_t>(i * n + j)] +=
                        -2.0 * x[static_cast<std::size_t>(i)] * y[static_cast<std::size_t>(j)];

            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                std::vector<double> stored = in_layout(a, m, n, layout);
                const int64_t lda = (layout == Core::Layout::RowMajor) ? n : m;

                Kernels::Reference::ger(layout, m, n, -2.0, x.data(), 1, y.data(), 1,
                                        stored.data(), lda);

                EXPECT_EQ(stored, in_layout(expected, m, n, layout));
            }
        }

        TEST(Small_ReferenceGer, TheConjugatedFormConjugatesY)
        {
            const std::vector<std::complex<double>> x{{1.0, 1.0}};
            const std::vector<std::complex<double>> y{{0.0, 2.0}};
            std::vector<std::complex<double>> a{{0.0, 0.0}};

            Kernels::Reference::ger<std::complex<double>, Kernels::Conjugated>(
                Core::Layout::RowMajor, 1, 1, std::complex<double>(1.0, 0.0), x.data(), 1,
                y.data(), 1, a.data(), 1);

            EXPECT_EQ(a[0], std::complex<double>(2.0, -2.0));
        }
    } // namespace Test
} // namespace SushiBLAS
