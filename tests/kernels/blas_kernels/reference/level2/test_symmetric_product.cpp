/**************************************************************************/
/* test_symmetric_product.cpp                                             */
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
 * @file test_symmetric_product.cpp
 * @brief Tests the reference symmetric product through a packed and a band access.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_seam_kernels/band_access.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/blas_seam_kernels/packed_access.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/symmetric_product.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_ReferenceSymmetricProduct, APackedAccessReadsTheWholeSymmetricMatrix)
        {
            const int64_t n = 5;
            const std::vector<double> a = small_integers<double>(n * n, 1);
            const std::vector<double> x = small_integers<double>(n, 2);
            for (const bool upper : {false, true})
            {
                SCOPED_TRACE(upper);
                std::vector<double> y = small_integers<double>(n, 3);
                const std::vector<double> expected =
                    dense_matvec(symmetric_from(a, n, upper), n, n, false, 2.0, x, 1.0, y);
                const std::vector<double> packed =
                    to_packed(a, n, uplo_of(upper), Core::Layout::ColumnMajor);
                const Kernels::PackedAccess access = Kernels::packed_access(
                    Core::Layout::ColumnMajor, uplo_of(upper), Core::Transpose::NoTrans, n);

                Kernels::Reference::Detail::symmetric_product<double, Kernels::Unconjugated>(
                    access, n, 2.0, packed.data(), x.data(), 1, 1.0, y.data(), 1);

                EXPECT_EQ(y, expected);
            }
        }

        TEST(Small_ReferenceSymmetricProduct, ABandAccessReadsOnlyTheBand)
        {
            const int64_t n = 6;
            const int64_t k = 1;
            const int64_t ld = k + 1;
            const std::vector<double> a =
                band_of(symmetric_from(small_integers<double>(n * n, 4), n, false), n, n, k, k);
            const std::vector<double> x = small_integers<double>(n, 5);
            std::vector<double> y(static_cast<std::size_t>(n), 0.0);
            const std::vector<double> expected = dense_matvec(a, n, n, false, 1.0, x, 0.0, y);
            const std::vector<double> band =
                to_triangle_band(a, n, Core::Uplo::Lower, k, ld, Core::Layout::RowMajor, 99.0);
            const Kernels::BandAccess access = Kernels::triangle_band_access(
                Core::Layout::RowMajor, Core::Uplo::Lower, Core::Transpose::NoTrans, k, ld);

            Kernels::Reference::Detail::symmetric_product<double, Kernels::Unconjugated>(
                access, n, 1.0, band.data(), x.data(), 1, 0.0, y.data(), 1);

            EXPECT_EQ(y, expected);
        }
    } // namespace Test
} // namespace SushiBLAS
