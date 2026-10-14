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

#include <vector>
#include <cstddef>
#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/substitution/running_solution.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>

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
            void trsm(Core::Layout layout, Core::Side side, Core::Uplo uplo, Core::Transpose trans,
                      Core::Diag diag, int64_t m, int64_t n, T alpha,
                      const T* a, int64_t lda,
                      T* b, int64_t ldb)
            {
                using W = Accumulator<T>;
                using RW = ReferenceAccumulator<T>;
                const bool left = (side == Core::Side::Left);
                const bool unit_diag = (diag == Core::Diag::Unit);
                const bool op_upper = Core::is_transposed(trans) != (uplo == Core::Uplo::Upper);
                const OperandAccess op_a = operand_access(layout, trans, lda);
                const OperandAccess at_b = operand_access(layout, Core::Transpose::NoTrans, ldb);

                // The workspace is indexed like B, so it reaches as far as B's last element.
                const bool needs_wide = RunningSolution<T>::K_NEEDS_WORKSPACE && m > 0 && n > 0;
                std::vector<W> wide(needs_wide
                                        ? static_cast<std::size_t>(at_b.index(m - 1, n - 1) + 1)
                                        : std::size_t{0});
                const RunningSolution<T> X(b, wide.data(), 0);

                if (left)
                {
                    for (int64_t j = 0; j < n; ++j)
                    {
                        if (!op_upper)
                        {
                            for (int64_t i = 0; i < m; ++i)
                            {
                                ReferenceSum<T> sum;
                                sum.add(RW(alpha) * RW(b[at_b.index(i, j)]));
                                for (int64_t k = 0; k < i; ++k)
                                    sum.add(-(RW(op_a.at(a, i, k)) * RW(X.get(at_b.index(k, j)))));
                                X.set(at_b.index(i, j),
                                      static_cast<W>(unit_diag
                                                         ? sum.value()
                                                         : sum.value() / RW(op_a.at(a, i, i))));
                            }
                        }
                        else
                        {
                            for (int64_t i = m - 1; i >= 0; --i)
                            {
                                ReferenceSum<T> sum;
                                sum.add(RW(alpha) * RW(b[at_b.index(i, j)]));
                                for (int64_t k = i + 1; k < m; ++k)
                                    sum.add(-(RW(op_a.at(a, i, k)) * RW(X.get(at_b.index(k, j)))));
                                X.set(at_b.index(i, j),
                                      static_cast<W>(unit_diag
                                                         ? sum.value()
                                                         : sum.value() / RW(op_a.at(a, i, i))));
                            }
                        }
                    }
                }
                else
                {
                    // Row i solves against op(A)^T; its element (p, q) is op(A)(q, p).
                    for (int64_t i = 0; i < m; ++i)
                    {
                        if (op_upper)
                        {
                            for (int64_t p = 0; p < n; ++p)
                            {
                                ReferenceSum<T> sum;
                                sum.add(RW(alpha) * RW(b[at_b.index(i, p)]));
                                for (int64_t q = 0; q < p; ++q)
                                    sum.add(-(RW(op_a.at(a, q, p)) * RW(X.get(at_b.index(i, q)))));
                                X.set(at_b.index(i, p),
                                      static_cast<W>(unit_diag
                                                         ? sum.value()
                                                         : sum.value() / RW(op_a.at(a, p, p))));
                            }
                        }
                        else
                        {
                            for (int64_t p = n - 1; p >= 0; --p)
                            {
                                ReferenceSum<T> sum;
                                sum.add(RW(alpha) * RW(b[at_b.index(i, p)]));
                                for (int64_t q = p + 1; q < n; ++q)
                                    sum.add(-(RW(op_a.at(a, q, p)) * RW(X.get(at_b.index(i, q)))));
                                X.set(at_b.index(i, p),
                                      static_cast<W>(unit_diag
                                                         ? sum.value()
                                                         : sum.value() / RW(op_a.at(a, p, p))));
                            }
                        }
                    }
                }
            }

            /**
             * @brief Solves a batch of triangular systems with the reference kernel.
             * @param str_a Element stride between consecutive A matrices; str_b likewise.
             * @see kernels/README.md
             */
            template<typename T>
            void trsm(Core::Layout layout, Core::Side side, Core::Uplo uplo, Core::Transpose trans,
                      Core::Diag diag, int64_t m, int64_t n, T alpha,
                      const T* a, int64_t lda, int64_t str_a,
                      T* b, int64_t ldb, int64_t str_b,
                      int64_t batch_size)
            {
                for (int64_t bidx = 0; bidx < batch_size; ++bidx)
                    trsm(layout, side, uplo, trans, diag, m, n, alpha,
                         a + bidx * str_a, lda, b + bidx * str_b, ldb);
            }

        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
