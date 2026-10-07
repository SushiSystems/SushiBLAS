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
 * @brief Defines the reference kernel of GEMM and its form over operand-access bricks.
 * @author Mustafa Garip
 */

#pragma once

#include <complex>
#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/reference_accumulator.hpp>
#include <SushiBLAS/kernels/blas/operand_access.hpp>
#include <SushiBLAS/kernels/blas/level3/gemm_epilogue.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes `C = Epilogue(alpha * X * Y + beta * C)` for operands read through
             *        access bricks.
             * @tparam AccessA Brick with `T at(const T*, int64_t, int64_t) const`; AccessB too.
             * @param beta Exactly zero skips reading C.
             * @see include/SushiBLAS/kernels/README.md
             */
            template<typename T, typename AccessA, typename AccessB,
                     typename Epilogue = GemmIdentityEpilogue>
            void gemm_accessed(int64_t m, int64_t n, int64_t k,
                               T alpha, const T* a, AccessA a_access,
                               const T* b, AccessB b_access,
                               T beta, T* c, OperandAccess c_access,
                               Epilogue epilogue = Epilogue{})
            {
                using RW = ReferenceAccumulator<T>;

                for (int64_t i = 0; i < m; ++i)
                {
                    for (int64_t j = 0; j < n; ++j)
                    {
                        ReferenceSum<T> sum;
                        for (int64_t l = 0; l < k; ++l)
                            sum.add(RW(a_access.at(a, i, l)) * RW(b_access.at(b, l, j)));

                        const int64_t idx = c_access.index(i, j);

                        // beta == 0 must not read the prior C value, which may be uninitialised; see include/SushiBLAS/kernels/README.md.
                        const T value = (beta == T(0))
                            ? T(RW(alpha) * sum.value())
                            : T(RW(alpha) * sum.value() + RW(beta) * RW(c[idx]));

                        c[idx] = epilogue(value, i, j);
                    }
                }
            }

            /**
             * @brief Computes a batch of products through access bricks with the reference kernel.
             * @param str_a Element stride between consecutive A matrices; str_b and str_c likewise.
             * @param epilogue Receives the row and column within one matrix of the batch.
             * @see include/SushiBLAS/kernels/README.md
             */
            template<typename T, typename AccessA, typename AccessB,
                     typename Epilogue = GemmIdentityEpilogue>
            void gemm_accessed(int64_t m, int64_t n, int64_t k,
                               T alpha, const T* a, AccessA a_access, int64_t str_a,
                               const T* b, AccessB b_access, int64_t str_b,
                               T beta, T* c, OperandAccess c_access, int64_t str_c,
                               int64_t batch_size,
                               Epilogue epilogue = Epilogue{})
            {
                for (int64_t b_idx = 0; b_idx < batch_size; ++b_idx)
                {
                    gemm_accessed(m, n, k, alpha,
                                  a + b_idx * str_a, a_access,
                                  b + b_idx * str_b, b_access,
                                  beta, c + b_idx * str_c, c_access, epilogue);
                }
            }

            /**
             * @brief Computes GEMM in pure C++ for row-major and column-major operands.
             * @tparam Epilogue Functor applied per element before the store, as `(value, row, col)`.
             * @param beta Exactly zero skips reading C.
             * @see include/SushiBLAS/kernels/README.md
             */
            template<typename T, typename Epilogue = GemmIdentityEpilogue>
            void gemm(Core::Layout layout, Core::Transpose transA, Core::Transpose transB,
                      int64_t m, int64_t n, int64_t k,
                      T alpha, const T* a, int64_t lda,
                      const T* b, int64_t ldb,
                      T beta, T* c, int64_t ldc,
                      Epilogue epilogue = Epilogue{})
            {
                gemm_accessed(m, n, k,
                              alpha, a, operand_access(layout, transA, lda),
                              b, operand_access(layout, transB, ldb),
                              beta, c, operand_access(layout, Core::Transpose::NoTrans, ldc),
                              epilogue);
            }

            /**
             * @brief Computes a batch of GEMMs with the reference kernel.
             * @param beta Exactly zero skips reading C.
             * @param str_a Element stride between consecutive A matrices; str_b and str_c likewise.
             * @param epilogue Receives the row and column within one matrix of the batch.
             * @see include/SushiBLAS/kernels/README.md
             */
            template<typename T, typename Epilogue = GemmIdentityEpilogue>
            void gemm(Core::Layout layout, Core::Transpose transA, Core::Transpose transB,
                      int64_t m, int64_t n, int64_t k,
                      T alpha, const T* a, int64_t lda, int64_t str_a,
                      const T* b, int64_t ldb, int64_t str_b,
                      T beta, T* c, int64_t ldc, int64_t str_c,
                      int64_t batch_size,
                      Epilogue epilogue = Epilogue{})
            {
                gemm_accessed(m, n, k,
                              alpha, a, operand_access(layout, transA, lda), str_a,
                              b, operand_access(layout, transB, ldb), str_b,
                              beta, c, operand_access(layout, Core::Transpose::NoTrans, ldc),
                              str_c, batch_size, epilogue);
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
