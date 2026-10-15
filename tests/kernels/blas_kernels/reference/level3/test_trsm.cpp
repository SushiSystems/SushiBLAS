/**************************************************************************/
/* test_trsm.cpp                                                          */
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
 * @file test_trsm.cpp
 * @brief Tests that the reference trsm recovers X from op(A) * X or X * op(A), single and batched.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level3/trsm.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The rows of B.
        constexpr int64_t K_ROWS = 4;

        /// @brief The columns of B.
        constexpr int64_t K_COLUMNS = 3;

        /** @brief Returns the product of two dense matrices given row by row. */
        static std::vector<double> product(const std::vector<double>& p, int64_t rows,
                                           int64_t inner, const std::vector<double>& q,
                                           int64_t columns)
        {
            std::vector<double> out(static_cast<std::size_t>(rows * columns), 0.0);
            for (int64_t i = 0; i < rows; ++i)
                for (int64_t j = 0; j < columns; ++j)
                    for (int64_t k = 0; k < inner; ++k)
                        out[static_cast<std::size_t>(i * columns + j)] +=
                            element(p, inner, i, k) * element(q, columns, k, j);
            return out;
        }

        /** @brief Returns the transpose of an n x n dense matrix. */
        static std::vector<double> transpose(const std::vector<double>& square, int64_t n)
        {
            std::vector<double> out(square.size());
            for (int64_t i = 0; i < n; ++i)
                for (int64_t j = 0; j < n; ++j)
                    out[static_cast<std::size_t>(j * n + i)] = element(square, n, i, j);
            return out;
        }

        TEST(Unit_ReferenceTrsm, RecoversXForEverySideTriangleAndTranspose)
        {
            const std::vector<double> x = small_integers<double>(K_ROWS * K_COLUMNS, 1);
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                for (const bool left : {true, false})
                {
                    const int64_t order = left ? K_ROWS : K_COLUMNS;
                    // A unit diagonal makes the solve exact in integers.
                    const std::vector<double> a = unit_diagonal_matrix<double>(order, 2);
                    for (const bool upper : {false, true})
                    {
                        for (const bool transposed : {false, true})
                        {
                            SCOPED_TRACE(4 * left + 2 * upper + transposed);
                            std::vector<double> op_a = triangle_of(a, order, upper, false);
                            if (transposed)
                                op_a = transpose(op_a, order);
                            const std::vector<double> b =
                                left ? product(op_a, K_ROWS, K_ROWS, x, K_COLUMNS)
                                     : product(x, K_ROWS, K_COLUMNS, op_a, K_COLUMNS);
                            std::vector<double> stored = in_layout(b, K_ROWS, K_COLUMNS, layout);
                            const int64_t ldb =
                                (layout == Core::Layout::RowMajor) ? K_COLUMNS : K_ROWS;

                            Kernels::Reference::trsm(
                                layout, side_of(left), uplo_of(upper), transpose_of(transposed),
                                Core::Diag::NonUnit, K_ROWS, K_COLUMNS, 1.0,
                                in_layout(a, order, order, layout).data(), order, stored.data(),
                                ldb);

                            EXPECT_EQ(stored, in_layout(x, K_ROWS, K_COLUMNS, layout));
                        }
                    }
                }
            }
        }

        TEST(Unit_ReferenceTrsm, AlphaScalesTheRightHandSide)
        {
            const std::vector<double> a{1.0, 0.0, 2.0, 1.0};
            std::vector<double> b{2.0, 6.0};

            Kernels::Reference::trsm(Core::Layout::RowMajor, Core::Side::Left, Core::Uplo::Lower,
                                     Core::Transpose::NoTrans, Core::Diag::Unit, 2, 1, 0.5,
                                     a.data(), 2, b.data(), 1);

            EXPECT_EQ(b, (std::vector<double>{1.0, 1.0}));
        }

        TEST(Unit_ReferenceTrsm, TheBatchedFormSolvesEachSystemOfTheBatch)
        {
            const int64_t batch = 2;
            const std::vector<double> a = unit_diagonal_matrix<double>(K_ROWS, 3);
            std::vector<double> batched = small_integers<double>(batch * K_ROWS * K_COLUMNS, 4);
            std::vector<double> one_by_one = batched;

            Kernels::Reference::trsm(Core::Layout::RowMajor, Core::Side::Left, Core::Uplo::Upper,
                                     Core::Transpose::NoTrans, Core::Diag::Unit, K_ROWS,
                                     K_COLUMNS, 1.0, a.data(), K_ROWS, 0, batched.data(),
                                     K_COLUMNS, K_ROWS * K_COLUMNS, batch);
            for (int64_t i = 0; i < batch; ++i)
                Kernels::Reference::trsm(Core::Layout::RowMajor, Core::Side::Left,
                                         Core::Uplo::Upper, Core::Transpose::NoTrans,
                                         Core::Diag::Unit, K_ROWS, K_COLUMNS, 1.0, a.data(),
                                         K_ROWS, one_by_one.data() + i * K_ROWS * K_COLUMNS,
                                         K_COLUMNS);

            EXPECT_EQ(batched, one_by_one);
        }
    } // namespace Test
} // namespace SushiBLAS
