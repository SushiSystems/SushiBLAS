/**************************************************************************/
/* test_triangular_product.cpp                                            */
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
 * @file test_triangular_product.cpp
 * @brief Tests the reference triangular product through a packed access, for every flag.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_seam_kernels/packed_access.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/triangular_product.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_ReferenceTriangularProduct, MatchesADenseProductThroughAPackedAccess)
        {
            const int64_t n = 5;
            const std::vector<double> a = small_integers<double>(n * n, 1);
            const std::vector<double> start = small_integers<double>(n, 2);
            for (const bool upper : {false, true})
            {
                for (const bool transposed : {false, true})
                {
                    for (const bool unit : {false, true})
                    {
                        SCOPED_TRACE(4 * upper + 2 * transposed + unit);
                        std::vector<double> x = start;
                        const std::vector<double> expected =
                            dense_matvec(triangle_of(a, n, upper, unit), n, n, transposed, 1.0,
                                         start, 0.0, start);
                        const std::vector<double> packed =
                            to_packed(a, n, uplo_of(upper), Core::Layout::RowMajor);
                        const Kernels::PackedAccess access = Kernels::packed_access(
                            Core::Layout::RowMajor, uplo_of(upper), transpose_of(transposed), n);

                        Kernels::Reference::Detail::triangular_product(
                            access, transposed != upper, unit, n, packed.data(), x.data(), 1);

                        EXPECT_EQ(x, expected);
                    }
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
