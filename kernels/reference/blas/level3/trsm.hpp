/****************************************************************/
/* trsm.hpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// Also backs Level2::trsv (a single-RHS-column TRSM: n=1). See
// src/engine/blas/level2/trsv.cpp.

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
            * @brief Pure C++ reference TRSM: solve op(A)*X=alpha*B (left_side) or
            * X*op(A)=alpha*B (right_side) for X, overwriting B, A triangular.
            *
            * Left side is solved one column of B at a time via forward (lower
            * effective triangle) or backward (upper) substitution against the
            * transpose-adjusted "effective" triangle of A - see
            * kernels/sycl/blas/level3/trsm.hpp for the derivation. Right side
            * is the equivalent row-wise system obtained by transposing once more
            * (X*op(A)=alpha*B  <=>  op(A)^T * X^T = alpha*B^T).
            */
            template<typename T>
            void trsm(Core::Layout layout, bool left_side, bool upper, bool transA, bool unit_diag,
                      int64_t m, int64_t n, T alpha,
                      const T* a, int64_t lda,
                      T* b, int64_t ldb)
            {
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
                                T sum = alpha * B_ref(i, j);
                                for (int64_t k = 0; k < i; ++k) sum -= A_eff(i, k) * B_ref(k, j);
                                B_ref(i, j) = unit_diag ? sum : sum / A_eff(i, i);
                            }
                        }
                        else
                        {
                            for (int64_t i = m - 1; i >= 0; --i)
                            {
                                T sum = alpha * B_ref(i, j);
                                for (int64_t k = i + 1; k < m; ++k) sum -= A_eff(i, k) * B_ref(k, j);
                                B_ref(i, j) = unit_diag ? sum : sum / A_eff(i, i);
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
                                T sum = alpha * B_ref(i, p);
                                for (int64_t q = 0; q < p; ++q) sum -= A_eff(p, q) * B_ref(i, q);
                                B_ref(i, p) = unit_diag ? sum : sum / A_eff(p, p);
                            }
                        }
                        else
                        {
                            for (int64_t p = n - 1; p >= 0; --p)
                            {
                                T sum = alpha * B_ref(i, p);
                                for (int64_t q = p + 1; q < n; ++q) sum -= A_eff(p, q) * B_ref(i, q);
                                B_ref(i, p) = unit_diag ? sum : sum / A_eff(p, p);
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
