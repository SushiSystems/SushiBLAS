/**************************************************************************/
/* syrk.hpp                                                               */
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
 * @file syrk.hpp
 * @brief Defines the reference kernels of SYRK, single and batched.
 * @author Mustafa Garip
 */

#pragma once

#include <complex>
#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/support/accumulator.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes SYRK, C = alpha*op(A)*op(A)^T + beta*C, on one triangle of C.
             * @see kernels/README.md
             */
            template<typename T>
            void syrk(Core::Layout layout, bool upper, bool trans,
                      int64_t n, int64_t k,
                      T alpha, const T* a, int64_t lda,
                      T beta, T* c, int64_t ldc)
            {
                using W = Accumulator<T>;
                const bool col_major = (layout == Core::Layout::COLUMN_MAJOR);

                // Element (r, col) of the logical, post-transpose op(A).
                auto A_at = [&](int64_t r, int64_t col) -> T
                {
                    return (trans != col_major) ? a[r + col * lda] : a[r * lda + col];
                };

                for (int64_t i = 0; i < n; ++i)
                {
                    for (int64_t j = 0; j < n; ++j)
                    {
                        if ((upper && j < i) || (!upper && j > i))
                            continue;

                        // The second factor is op(A)^T(l, j), which is op(A)(j, l).
                        W sum(0);
                        for (int64_t l = 0; l < k; ++l)
                            sum += W(A_at(i, l)) * W(A_at(j, l));

                        // A zero beta must not read the prior value of C.
                        T& out = (layout == Core::Layout::ROW_MAJOR) ? c[i * ldc + j] : c[i + j * ldc];
                        out = (beta == T(0)) ? T(W(alpha) * sum) : T(W(alpha) * sum + W(beta) * W(out));
                    }
                }
            }

            /**
            * @brief Batch SYRK reference implementation.
            */
            template<typename T>
            void syrk(Core::Layout layout, bool upper, bool trans,
                      int64_t n, int64_t k,
                      T alpha, const T* a, int64_t lda, int64_t str_a,
                      T beta, T* c, int64_t ldc, int64_t str_c,
                      int64_t batch_size) 
            {
                for (int64_t b_idx = 0; b_idx < batch_size; ++b_idx) 
                {
                    syrk(layout, upper, trans, n, k, alpha, 
                         a + b_idx * str_a, lda, 
                         beta, c + b_idx * str_c, ldc);
                }
            }

        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
