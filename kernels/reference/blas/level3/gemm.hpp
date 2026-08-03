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
            */
            template<typename T>
            void gemm(Core::Layout layout, bool transA, bool transB,
                    int64_t m, int64_t n, int64_t k,
                    T alpha, const T* a, int64_t lda,
                    const T* b, int64_t ldb,
                    T beta, T* c, int64_t ldc)
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
            * @brief Batch GEMM reference implementation.
            */
            template<typename T>
            void gemm(Core::Layout layout, bool transA, bool transB,
                            int64_t m, int64_t n, int64_t k,
                            T alpha, const T* a, int64_t lda, int64_t str_a,
                            const T* b, int64_t ldb, int64_t str_b,
                            T beta, T* c, int64_t ldc, int64_t str_c,
                            int64_t batch_size) 
            {
                for (int64_t b_idx = 0; b_idx < batch_size; ++b_idx) 
                {
                    gemm(layout, transA, transB, m, n, k, alpha, 
                        a + b_idx * str_a, lda, 
                        b + b_idx * str_b, ldb, 
                        beta, c + b_idx * str_c, ldc);
                }
            }

        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
