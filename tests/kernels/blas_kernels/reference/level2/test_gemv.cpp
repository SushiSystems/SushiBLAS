/**************************************************************************/
/* test_gemv.cpp                                                          */
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
 * @file test_gemv.cpp
 * @brief Tests the reference gemv against a dense loop, in both layouts and transposes.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/gemv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Rows of the matrix.
        constexpr int64_t K_ROWS = 4;

        /// @brief Columns of the matrix.
        constexpr int64_t K_COLUMNS = 5;

        TEST(Unit_ReferenceGemv, MatchesADenseLoopInEveryLayoutAndTranspose)
        {
            const std::vector<double> a = small_integers<double>(K_ROWS * K_COLUMNS, 1);
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                for (const bool transposed : {false, true})
                {
                    SCOPED_TRACE(transposed);
                    const int64_t in_length = transposed ? K_ROWS : K_COLUMNS;
                    const int64_t out_length = transposed ? K_COLUMNS : K_ROWS;
                    const std::vector<double> x = small_integers<double>(in_length, 2);
                    std::vector<double> y = small_integers<double>(out_length, 3);
                    const std::vector<double> expected =
                        dense_matvec(a, K_ROWS, K_COLUMNS, transposed, 2.0, x, -1.0, y);
                    const int64_t lda = (layout == Core::Layout::RowMajor) ? K_COLUMNS : K_ROWS;

                    Kernels::Reference::gemv(layout, transpose_of(transposed), K_ROWS, K_COLUMNS,
                                             2.0, in_layout(a, K_ROWS, K_COLUMNS, layout).data(),
                                             lda, x.data(), 1, -1.0, y.data(), 1);

                    EXPECT_EQ(y, expected);
                }
            }
        }

        TEST(Unit_ReferenceGemv, AZeroBetaIgnoresANaNInY)
        {
            const std::vector<double> a{1.0, 2.0, 3.0, 4.0};
            const std::vector<double> x{1.0, 1.0};
            std::vector<double> y{std::nan(""), std::nan("")};

            Kernels::Reference::gemv(Core::Layout::RowMajor, Core::Transpose::NoTrans, 2, 2, 1.0,
                                     a.data(), 2, x.data(), 1, 0.0, y.data(), 1);

            EXPECT_EQ(y, (std::vector<double>{3.0, 7.0}));
        }

        TEST(Unit_ReferenceGemv, StridesSkipTheElementsBetween)
        {
            const std::vector<double> a{1.0, 2.0, 3.0, 4.0};
            const std::vector<double> x{1.0, -9.0, 2.0};
            std::vector<double> y{0.0, -9.0, 0.0};

            Kernels::Reference::gemv(Core::Layout::RowMajor, Core::Transpose::NoTrans, 2, 2, 1.0,
                                     a.data(), 2, x.data(), 2, 0.0, y.data(), 2);

            EXPECT_EQ(y, (std::vector<double>{5.0, -9.0, 11.0}));
        }
    } // namespace Test
} // namespace SushiBLAS
