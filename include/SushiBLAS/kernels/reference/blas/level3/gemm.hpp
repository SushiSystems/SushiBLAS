/**************************************************************************/
/* gemm.hpp                                                               */
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
 * @file gemm.hpp
 * @brief Declares the pure-C++ reference GEMM kernels.
 * @author Mustafa Garip
 */

#pragma once

#include <complex>
#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/kernels/blas/level3/gemm_epilogue.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes GEMM in pure C++ for row-major and column-major operands.
             * @tparam Epilogue Functor applied per element before the store, as `(value, row, col)`.
             * @param beta Exactly zero skips reading C.
             * @see include/SushiBLAS/kernels/README.md
             */
            template<typename T, typename Epilogue = GemmIdentityEpilogue>
            void gemm(Core::Layout layout, bool transA, bool transB,
                    int64_t m, int64_t n, int64_t k,
                    T alpha, const T* a, int64_t lda,
                    const T* b, int64_t ldb,
                    T beta, T* c, int64_t ldc,
                    Epilogue epilogue = Epilogue{})
            {
                using W = Accumulator<T>;
                const bool col_major = (layout == Core::Layout::COLUMN_MAJOR);

                // Element (r, col) of the logical, post-transpose operand.
                auto A_at = [&](int64_t r, int64_t col) -> T
                {
                    return (transA != col_major) ? a[r + col * lda] : a[r * lda + col];
                };
                auto B_at = [&](int64_t r, int64_t col) -> T
                {
                    return (transB != col_major) ? b[r + col * ldb] : b[r * ldb + col];
                };

                for (int64_t i = 0; i < m; ++i)
                {
                    for (int64_t j = 0; j < n; ++j)
                    {
                        W sum(0);
                        for (int64_t l = 0; l < k; ++l)
                            sum += W(A_at(i, l)) * W(B_at(l, j));

                        const int64_t idx = (layout == Core::Layout::ROW_MAJOR)
                            ? i * ldc + j
                            : i + j * ldc;

                        // beta == 0 must not read the prior C value, which may be uninitialised; see include/SushiBLAS/kernels/README.md.
                        const T value = (beta == T(0)) ? T(W(alpha) * sum)
                                                       : T(W(alpha) * sum + W(beta) * W(c[idx]));

                        c[idx] = epilogue(value, i, j);
                    }
                }
            }

            /**
             * @brief Computes a batch of GEMMs with the reference kernel.
             * @param beta Exactly zero skips reading C.
             * @param str_a Element stride between consecutive A matrices; str_b and str_c likewise.
             * @param epilogue Receives the row and column within one matrix of the batch.
             * @see include/SushiBLAS/kernels/README.md
             */
            template<typename T, typename Epilogue = GemmIdentityEpilogue>
            void gemm(Core::Layout layout, bool transA, bool transB,
                            int64_t m, int64_t n, int64_t k,
                            T alpha, const T* a, int64_t lda, int64_t str_a,
                            const T* b, int64_t ldb, int64_t str_b,
                            T beta, T* c, int64_t ldc, int64_t str_c,
                            int64_t batch_size,
                            Epilogue epilogue = Epilogue{})
            {
                for (int64_t b_idx = 0; b_idx < batch_size; ++b_idx)
                {
                    gemm(layout, transA, transB, m, n, k, alpha,
                        a + b_idx * str_a, lda,
                        b + b_idx * str_b, ldb,
                        beta, c + b_idx * str_c, ldc, epilogue);
                }
            }

        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
