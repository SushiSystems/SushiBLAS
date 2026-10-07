/**************************************************************************/
/* trsm.hpp                                                               */
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
 * @file trsm.hpp
 * @brief Defines the reference kernels of TRSM, single and batched.
 * @author Mustafa Garip
 */

#pragma once

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
             * @brief Solves op(A)*X = alpha*B or X*op(A) = alpha*B for X, overwriting B.
             * @see kernels/README.md
             */
            template<typename T>
            void trsm(Core::Layout layout, bool left_side, bool upper, bool transA, bool unit_diag,
                      int64_t m, int64_t n, T alpha,
                      const T* a, int64_t lda,
                      T* b, int64_t ldb)
            {
                using W = Accumulator<T>;
                auto A_at = [&](int64_t r, int64_t c) -> T
                {
                    return (layout == Core::Layout::ROW_MAJOR) ? a[r * lda + c] : a[r + c * lda];
                };
                auto B_ref = [&](int64_t r, int64_t c) -> T&
                {
                    return (layout == Core::Layout::ROW_MAJOR) ? b[r * ldb + c] : b[r + c * ldb];
                };

                if (left_side)
                {
                    bool eff_lower = transA ? upper : !upper;
                    auto A_eff = [&](int64_t i, int64_t k) -> T { return transA ? A_at(k, i) : A_at(i, k); };

                    for (int64_t j = 0; j < n; ++j)
                    {
                        if (eff_lower)
                        {
                            for (int64_t i = 0; i < m; ++i)
                            {
                                W sum = W(alpha) * W(B_ref(i, j));
                                for (int64_t k = 0; k < i; ++k) sum -= W(A_eff(i, k)) * W(B_ref(k, j));
                                B_ref(i, j) = unit_diag ? T(sum) : T(sum / W(A_eff(i, i)));
                            }
                        }
                        else
                        {
                            for (int64_t i = m - 1; i >= 0; --i)
                            {
                                W sum = W(alpha) * W(B_ref(i, j));
                                for (int64_t k = i + 1; k < m; ++k) sum -= W(A_eff(i, k)) * W(B_ref(k, j));
                                B_ref(i, j) = unit_diag ? T(sum) : T(sum / W(A_eff(i, i)));
                            }
                        }
                    }
                }
                else
                {
                    bool transA_row = !transA;
                    bool eff_lower = transA_row ? upper : !upper;
                    auto A_eff = [&](int64_t p, int64_t q) -> T { return transA_row ? A_at(q, p) : A_at(p, q); };

                    for (int64_t i = 0; i < m; ++i)
                    {
                        if (eff_lower)
                        {
                            for (int64_t p = 0; p < n; ++p)
                            {
                                W sum = W(alpha) * W(B_ref(i, p));
                                for (int64_t q = 0; q < p; ++q) sum -= W(A_eff(p, q)) * W(B_ref(i, q));
                                B_ref(i, p) = unit_diag ? T(sum) : T(sum / W(A_eff(p, p)));
                            }
                        }
                        else
                        {
                            for (int64_t p = n - 1; p >= 0; --p)
                            {
                                W sum = W(alpha) * W(B_ref(i, p));
                                for (int64_t q = p + 1; q < n; ++q) sum -= W(A_eff(p, q)) * W(B_ref(i, q));
                                B_ref(i, p) = unit_diag ? T(sum) : T(sum / W(A_eff(p, p)));
                            }
                        }
                    }
                }
            }

            /**
            * @brief Batched TRSM reference implementation.
            */
            template<typename T>
            void trsm(Core::Layout layout, bool left_side, bool upper, bool transA, bool unit_diag,
                      int64_t m, int64_t n, T alpha,
                      const T* a, int64_t lda, int64_t str_a,
                      T* b, int64_t ldb, int64_t str_b,
                      int64_t batch_size)
            {
                for (int64_t bidx = 0; bidx < batch_size; ++bidx)
                    trsm(layout, left_side, upper, transA, unit_diag, m, n, alpha,
                         a + bidx * str_a, lda, b + bidx * str_b, ldb);
            }

        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
