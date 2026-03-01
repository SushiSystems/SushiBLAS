/**************************************************************************/
/* gemm.hpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
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
            */
            template<typename T>
            void gemm(Core::Layout layout, bool transA, bool transB,
                    int64_t m, int64_t n, int64_t k,
                    T alpha, const T* a, int64_t lda,
                    const T* b, int64_t ldb,
                    T beta, T* c, int64_t ldc) 
            {
                for (int64_t i = 0; i < m; ++i) 
                {
                    for (int64_t j = 0; j < n; ++j) 
                    {
                        T sum(0);
                        for (int64_t l = 0; l < k; ++l) 
                        {
                            T valA, valB;
                            if (layout == Core::Layout::ROW_MAJOR) 
                            {
                                valA = transA ? a[l * lda + i] : a[i * lda + l];
                                valB = transB ? b[j * ldb + l] : b[l * ldb + j];
                            } else 
                            {
                                valA = transA ? a[i + l * lda] : a[l + i * lda];
                                valB = transB ? b[l + j * ldb] : b[j + l * ldb];
                            }
                            sum += valA * valB;
                        }

                        if (layout == Core::Layout::ROW_MAJOR) 
                            c[i * ldc + j] = alpha * sum + beta * c[i * ldc + j];
                        else 
                            c[i + j * ldc] = alpha * sum + beta * c[i + j * ldc];
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
