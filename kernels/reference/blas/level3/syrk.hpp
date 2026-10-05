/****************************************************************/
/* syrk.hpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#pragma once

#include <complex>
#include <cstdint>
#include <SushiBLAS/core/common.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
            * @brief Pure C++ reference implementation of SYRK.
            *
            * Supports Row-Major and Column-Major layouts.
            *
            * Operand addressing follows the same two-stride rule as
            * `Reference::gemm`, and used to carry the same defect: the COLUMN_MAJOR arm
            * was inverted, which made `layout` a no-op for A and read past its end for
            * some shapes. See that file for the identity that catches it.
            */
            template<typename T>
            void syrk(Core::Layout layout, bool upper, bool trans,
                      int64_t n, int64_t k,
                      T alpha, const T* a, int64_t lda,
                      T beta, T* c, int64_t ldc)
            {
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

                        // C = alpha*op(A)*op(A)^T, so the second factor is
                        // op(A)^T(l, j), which is op(A)(j, l).
                        T sum(0);
                        for (int64_t l = 0; l < k; ++l)
                            sum += A_at(i, l) * A_at(j, l);

                        // beta==0 must not read the (possibly uninitialized) prior C value,
                        // matching standard BLAS semantics.
                        if (layout == Core::Layout::ROW_MAJOR)
                            c[i * ldc + j] = (beta == T(0)) ? alpha * sum : alpha * sum + beta * c[i * ldc + j];
                        else
                            c[i + j * ldc] = (beta == T(0)) ? alpha * sum : alpha * sum + beta * c[i + j * ldc];
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
