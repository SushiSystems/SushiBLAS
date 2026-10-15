/**************************************************************************/
/* test_syrk.cpp                                                          */
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
 * @file test_syrk.cpp
 * @brief Tests the reference syrk, single and batched, against a dense rank-k update.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level3/syrk.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The order of C.
        constexpr int64_t K_ORDER = 4;

        /// @brief The inner dimension of the update.
        constexpr int64_t K_DEPTH = 3;

        /**
         * @brief Returns C with alpha * G * G^T + beta * C added on one triangle, G n x k.
         * @param g G given row by row.
         */
        static std::vector<double> rank_k(const std::vector<double>& g, std::vector<double> c,
                                   bool upper, double alpha, double beta)
        {
            for (int64_t i = 0; i < K_ORDER; ++i)
            {
                for (int64_t j = 0; j < K_ORDER; ++j)
                {
                    if (!(upper ? (i <= j) : (i >= j)))
                        continue;
                    double sum = 0.0;
                    for (int64_t p = 0; p < K_DEPTH; ++p)
                        sum += element(g, K_DEPTH, i, p) * element(g, K_DEPTH, j, p);
                    double& entry = c[static_cast<std::size_t>(i * K_ORDER + j)];
                    entry = alpha * sum + beta * entry;
                }
            }
            return c;
        }

        TEST(Small_ReferenceSyrk, MatchesADenseUpdateForEveryLayoutTriangleAndTranspose)
        {
            const std::vector<double> g = small_integers<double>(K_ORDER * K_DEPTH, 1);
            const std::vector<double> c = small_integers<double>(K_ORDER * K_ORDER, 2);
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                for (const bool upper : {false, true})
                {
                    for (const bool transposed : {false, true})
                    {
                        SCOPED_TRACE(2 * upper + transposed);
                        // op(A) = G, so A is G itself or its transpose.
                        std::vector<double> a_dense = g;
                        int64_t rows = K_ORDER;
                        int64_t columns = K_DEPTH;
                        if (transposed)
                        {
                            for (int64_t i = 0; i < K_ORDER; ++i)
                                for (int64_t p = 0; p < K_DEPTH; ++p)
                                    a_dense[static_cast<std::size_t>(p * K_ORDER + i)] =
                                        element(g, K_DEPTH, i, p);
                            rows = K_DEPTH;
                            columns = K_ORDER;
                        }
                        const std::vector<double> a = in_layout(a_dense, rows, columns, layout);
                        const int64_t lda = (layout == Core::Layout::RowMajor) ? columns : rows;
                        std::vector<double> stored = in_layout(c, K_ORDER, K_ORDER, layout);

                        Kernels::Reference::syrk(layout, uplo_of(upper), transpose_of(transposed),
                                                 K_ORDER, K_DEPTH, 2.0, a.data(), lda, -1.0,
                                                 stored.data(), K_ORDER);

                        EXPECT_EQ(stored, in_layout(rank_k(g, c, upper, 2.0, -1.0), K_ORDER,
                                                    K_ORDER, layout));
                    }
                }
            }
        }

        TEST(Small_ReferenceSyrk, TheBatchedFormUpdatesEachMatrixOfTheBatch)
        {
            const int64_t batch = 2;
            const std::vector<double> a = small_integers<double>(batch * K_ORDER * K_DEPTH, 3);
            std::vector<double> batched = small_integers<double>(batch * K_ORDER * K_ORDER, 4);
            std::vector<double> one_by_one = batched;

            Kernels::Reference::syrk(Core::Layout::RowMajor, Core::Uplo::Lower,
                                     Core::Transpose::NoTrans, K_ORDER, K_DEPTH, 1.0, a.data(),
                                     K_DEPTH, K_ORDER * K_DEPTH, 1.0, batched.data(), K_ORDER,
                                     K_ORDER * K_ORDER, batch);
            for (int64_t b = 0; b < batch; ++b)
                Kernels::Reference::syrk(Core::Layout::RowMajor, Core::Uplo::Lower,
                                         Core::Transpose::NoTrans, K_ORDER, K_DEPTH, 1.0,
                                         a.data() + b * K_ORDER * K_DEPTH, K_DEPTH, 1.0,
                                         one_by_one.data() + b * K_ORDER * K_ORDER, K_ORDER);

            EXPECT_EQ(batched, one_by_one);
        }
    } // namespace Test
} // namespace SushiBLAS
