/****************************************************************/
/* trsm.hpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// Also backs Level2::trsv (a single-RHS-column TRSM: n=1). See
// src/engine/blas/level2/trsv.cpp.
//
// TRSM's substitution is inherently sequential along one matrix dimension,
// but each of the "n" columns (left side) or "m" rows (right side) is an
// independent triangular solve against the same A - no data hazard between
// them - so this parallelizes over that dimension, one work-item doing a
// full serial substitution per column/row. Derivation of the "effective
// triangle" (A_eff / eff_lower) used below:
//
//   Left side solves op(A)*X = alpha*B column by column. Define
//   A_eff(i,k) = transA ? A(k,i) : A(i,k) - this is exactly op(A)(i,k).
//   If A is upper and transA, op(A)=A^T is lower; if A is lower and transA,
//   op(A) is upper. So op(A) is lower ("eff_lower") iff transA XOR lower(A),
//   i.e. eff_lower = transA ? upper : !upper. Once eff_lower is known, the
//   standard forward (lower) / backward (upper) substitution over A_eff
//   applies directly, regardless of the original transpose/layout.
//
//   Right side solves X*op(A) = alpha*B, equivalent to op(A)^T * X^T =
//   alpha*B^T - i.e. the same left-side system with one more transpose
//   applied, so eff_lower uses transA_row = !transA in place of transA, and
//   substitution walks A's "p,q" index against B's row (fixed i, varying
//   column) instead of B's column.

#pragma once

#include <cstdint>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            template<typename T>
            [[nodiscard]] sycl::event trsm(sycl::queue& queue, Core::Layout layout,
                             bool left_side, bool upper, bool transA, bool unit_diag,
                             int64_t m, int64_t n, T alpha,
                             const T* a, int64_t lda, int64_t str_a,
                             T* b, int64_t ldb, int64_t str_b,
                             int64_t batch_size, const std::vector<sycl::event>& deps)
            {
                const int64_t num_vectors = left_side ? n : m;
                SB_LOG_INFO("SYCL TRSM: batch={} m={} n={}", batch_size, m, n);

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.parallel_for(sycl::range<2>(static_cast<size_t>(batch_size), static_cast<size_t>(num_vectors)),
                        [=](sycl::id<2> idx)
                        {
                            const int64_t batch_id = idx[0];
                            const int64_t vec_id = idx[1];
                            const T* A = a + batch_id * str_a;
                            T* B = b + batch_id * str_b;

                            auto A_at = [=](int64_t r, int64_t c) -> T
                            {
                                return (layout == Core::Layout::ROW_MAJOR) ? A[r * lda + c] : A[r + c * lda];
                            };
                            auto B_ref = [=](int64_t r, int64_t c) -> T&
                            {
                                return (layout == Core::Layout::ROW_MAJOR) ? B[r * ldb + c] : B[r + c * ldb];
                            };

                            if (left_side)
                            {
                                const int64_t j = vec_id;
                                bool eff_lower = transA ? upper : !upper;
                                auto A_eff = [=](int64_t i, int64_t k) -> T { return transA ? A_at(k, i) : A_at(i, k); };

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
                            else
                            {
                                const int64_t i = vec_id;
                                bool transA_row = !transA;
                                bool eff_lower = transA_row ? upper : !upper;
                                auto A_eff = [=](int64_t p, int64_t q) -> T { return transA_row ? A_at(q, p) : A_at(p, q); };

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
                        });
                });
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
