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

#include <vector>
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
            * @brief SYCL Dispatcher for GEMM operations.
            * 
            * Handles both single and batch GEMM using oneMKL.
            */
            template<typename T>
            [[nodiscard]] sycl::event gemm(sycl::queue& queue, Core::Layout layout,
                            oneapi::mkl::transpose transA, oneapi::mkl::transpose transB,
                            int64_t m, int64_t n, int64_t k,
                            T alpha, const T* a, int64_t lda, int64_t str_a,
                            const T* b, int64_t ldb, int64_t str_b,
                            T beta, T* c, int64_t ldc, int64_t str_c,
                            int64_t batch_size, const std::vector<sycl::event>& deps)
            {
                if (layout == Core::Layout::ROW_MAJOR) 
                {
                    if (batch_size > 1) 
                    {
                        SB_LOG_INFO("SYCL Batch GEMM [Row-Major]: {}x[{}x{}x{}]", batch_size, m, n, k);
                        return oneapi::mkl::blas::row_major::gemm_batch(queue, transA, transB, m, n, k, alpha, a, lda, str_a, b, ldb, str_b, beta, c, ldc, str_c, batch_size, deps);
                    } 
                    else 
                    {
                        SB_LOG_INFO("SYCL GEMM [Row-Major]: {}x{}x{}", m, n, k);
                        return oneapi::mkl::blas::row_major::gemm(queue, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, deps);
                    }
                } 
                else // COLUMN_MAJOR
                {
                    if (batch_size > 1) 
                    {
                        SB_LOG_INFO("SYCL Batch GEMM [Col-Major]: {}x[{}x{}x{}]", batch_size, m, n, k);
                        return oneapi::mkl::blas::column_major::gemm_batch(queue, transA, transB, m, n, k, alpha, a, lda, str_a, b, ldb, str_b, beta, c, ldc, str_c, batch_size, deps);
                    } 
                    else 
                    {
                        SB_LOG_INFO("SYCL GEMM [Col-Major]: {}x{}x{}", m, n, k);
                        return oneapi::mkl::blas::column_major::gemm(queue, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, deps);
                    }
                }
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
