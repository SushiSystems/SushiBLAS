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
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

#include <complex>
#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/kernels/blas/level3/gemm_epilogue.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
            * @brief Pure C++ reference implementation of GEMM.
            *
            * Supports Row-Major and Column-Major layouts.
            *
            * Addressing an operand is two strides, not four cases: transposing it and
            * flipping its layout are the same operation, so ROW_MAJOR+transposed and
            * COLUMN_MAJOR+untransposed index identically (as do the other pair). That
            * identity is the sanity check worth keeping in mind here - an earlier
            * version of this file had the COLUMN_MAJOR arm of A and B inverted, which
            * made `layout` a no-op for both operands and, with the column-major leading
            * dimension the driver passes, read past the end of A whenever m > k.
            *
            * @tparam T The element type.
            * @tparam Epilogue Functor applied per element immediately before the store,
            *         exactly as in @ref SushiBLAS::Kernels::SYCL::gemm and with the same
            *         `(value, row, col)` contract. It is here so that a *fused* GEMM has
            *         a ground truth at all: `VERIFY` mode and `FORCE_REFERENCE` are how
            *         this project judges a SYCL kernel, and a reference that dropped the
            *         epilogue would silently declare every fused result wrong. Defaults
            *         to @ref SushiBLAS::Kernels::GemmIdentityEpilogue.
            * @param layout The layout all three operands are stored in.
            * @param transA Whether A is used transposed.
            * @param transB Whether B is used transposed.
            * @param m Rows of C.
            * @param n Columns of C.
            * @param k Shared inner dimension.
            * @param alpha Scale applied to the product.
            * @param a Base pointer of A.
            * @param lda A's leading dimension.
            * @param b Base pointer of B.
            * @param ldb B's leading dimension.
            * @param beta Scale applied to the incoming C; exactly zero skips reading C.
            * @param c Base pointer of C.
            * @param ldc C's leading dimension.
            * @param epilogue The per-element epilogue functor.
            */
            template<typename T, typename Epilogue = GemmIdentityEpilogue>
            void gemm(Core::Layout layout, bool transA, bool transB,
                    int64_t m, int64_t n, int64_t k,
                    T alpha, const T* a, int64_t lda,
                    const T* b, int64_t ldb,
                    T beta, T* c, int64_t ldc,
                    Epilogue epilogue = Epilogue{})
            {
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
                        T sum(0);
                        for (int64_t l = 0; l < k; ++l)
                            sum += A_at(i, l) * B_at(l, j);

                        const int64_t idx = (layout == Core::Layout::ROW_MAJOR)
                            ? i * ldc + j
                            : i + j * ldc;

                        // beta==0 must not read the (possibly uninitialized) prior C value,
                        // matching standard BLAS semantics.
                        const T value = (beta == T(0)) ? alpha * sum
                                                       : alpha * sum + beta * c[idx];

                        c[idx] = epilogue(value, i, j);
                    }
                }
            }

            /**
            * @brief Batch GEMM reference implementation.
            *
            * @tparam T The element type.
            * @tparam Epilogue Functor applied per element before the store; see the
            *         single-matrix overload above.
            * @param layout The layout all three operands are stored in.
            * @param transA Whether A is used transposed.
            * @param transB Whether B is used transposed.
            * @param m Rows of C.
            * @param n Columns of C.
            * @param k Shared inner dimension.
            * @param alpha Scale applied to the product.
            * @param a Base pointer of A.
            * @param lda A's leading dimension.
            * @param str_a Element stride between consecutive A matrices in the batch.
            * @param b Base pointer of B.
            * @param ldb B's leading dimension.
            * @param str_b Element stride between consecutive B matrices in the batch.
            * @param beta Scale applied to the incoming C; exactly zero skips reading C.
            * @param c Base pointer of C.
            * @param ldc C's leading dimension.
            * @param str_c Element stride between consecutive C matrices in the batch.
            * @param batch_size Number of matrices in the batch.
            * @param epilogue The per-element epilogue functor. Its `row`/`col` arguments
            *        are the coordinates within one matrix of the batch, matching
            *        @ref SushiBLAS::Kernels::SYCL::gemm - a batch-aware epilogue would
            *        need a batch index the SYCL side does not supply either.
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
