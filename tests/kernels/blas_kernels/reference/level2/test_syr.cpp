/**************************************************************************/
/* test_syr.cpp                                                           */
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
 * @file test_syr.cpp
 * @brief Tests the reference syr, which adds alpha * x * x^T to one triangle and nothing else.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/syr.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_ReferenceSyr, UpdatesTheNamedTriangleOnly)
        {
            const int64_t n = 5;
            const std::vector<double> a = small_integers<double>(n * n, 1);
            const std::vector<double> x = small_integers<double>(n, 2);
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                for (const bool upper : {false, true})
                {
                    SCOPED_TRACE(upper);
                    std::vector<double> stored = in_layout(a, n, n, layout);
                    const std::vector<double> expected =
                        triangle_update(a, n, upper, 3.0, x, {}, false);

                    Kernels::Reference::syr(layout, uplo_of(upper), n, 3.0, x.data(), 1,
                                            stored.data(), n);

                    EXPECT_EQ(stored, in_layout(expected, n, n, layout));
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
