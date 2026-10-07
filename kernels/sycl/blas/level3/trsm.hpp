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
 * @brief Defines the SYCL kernel of TRSM.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/running_solution.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiBLAS/kernels/blas/operand_access.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Solves op(A)*X = alpha*B or X*op(A) = alpha*B for X, overwriting B.
             * @param work Holds batch_size * m * n elements when
             *        RunningSolution<T>::K_NEEDS_WORKSPACE is true; never read otherwise.
             * @pre When work is read, B has no padding: its last element is element m * n - 1.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event trsm(sycl::queue& queue, Core::Layout layout,
                             Core::Side side, Core::Uplo uplo, Core::Transpose trans,
                             Core::Diag diag, int64_t m, int64_t n, T alpha,
                             const T* a, int64_t lda, int64_t str_a,
                             T* b, int64_t ldb, int64_t str_b, int64_t batch_size,
                             Accumulator<T>* work, const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;
                const bool left = (side == Core::Side::LEFT);
                const bool unit_diag = (diag == Core::Diag::UNIT);
                const bool op_upper = Core::is_transposed(trans) != (uplo == Core::Uplo::UPPER);
                const OperandAccess op_a = operand_access(layout, trans, lda);
                const OperandAccess at_b = operand_access(layout, Core::Transpose::NO_TRANS, ldb);

                // The workspace is indexed like B, and holds m * n elements per matrix.
                SB_ASSERT(!RunningSolution<T>::K_NEEDS_WORKSPACE || m <= 0 || n <= 0 ||
                              at_b.index(m - 1, n - 1) < m * n,
                          "SYCL TRSM workspace cannot follow a padded B");

                const int64_t num_vectors = left ? n : m;
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
                            const RunningSolution<T> X(B, work, batch_id * m * n);

                            if (left)
                            {
                                const int64_t j = vec_id;
                                if (!op_upper)
                                {
                                    for (int64_t i = 0; i < m; ++i)
                                    {
                                        W sum = multiply(W(alpha), W(B[at_b.index(i, j)]));
                                        for (int64_t k = 0; k < i; ++k)
                                            sum -= multiply(W(op_a.at(A, i, k)),
                                                            X.get(at_b.index(k, j)));
                                        X.set(at_b.index(i, j),
                                              unit_diag ? sum : divide(sum, W(op_a.at(A, i, i))));
                                    }
                                }
                                else
                                {
                                    for (int64_t i = m - 1; i >= 0; --i)
                                    {
                                        W sum = multiply(W(alpha), W(B[at_b.index(i, j)]));
                                        for (int64_t k = i + 1; k < m; ++k)
                                            sum -= multiply(W(op_a.at(A, i, k)),
                                                            X.get(at_b.index(k, j)));
                                        X.set(at_b.index(i, j),
                                              unit_diag ? sum : divide(sum, W(op_a.at(A, i, i))));
                                    }
                                }
                            }
                            else
                            {
                                // Row i solves against op(A)^T; its element (p, q) is op(A)(q, p).
                                const int64_t i = vec_id;
                                if (op_upper)
                                {
                                    for (int64_t p = 0; p < n; ++p)
                                    {
                                        W sum = multiply(W(alpha), W(B[at_b.index(i, p)]));
                                        for (int64_t q = 0; q < p; ++q)
                                            sum -= multiply(W(op_a.at(A, q, p)),
                                                            X.get(at_b.index(i, q)));
                                        X.set(at_b.index(i, p),
                                              unit_diag ? sum : divide(sum, W(op_a.at(A, p, p))));
                                    }
                                }
                                else
                                {
                                    for (int64_t p = n - 1; p >= 0; --p)
                                    {
                                        W sum = multiply(W(alpha), W(B[at_b.index(i, p)]));
                                        for (int64_t q = p + 1; q < n; ++q)
                                            sum -= multiply(W(op_a.at(A, q, p)),
                                                            X.get(at_b.index(i, q)));
                                        X.set(at_b.index(i, p),
                                              unit_diag ? sum : divide(sum, W(op_a.at(A, p, p))));
                                    }
                                }
                            }
                        });
                });
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
