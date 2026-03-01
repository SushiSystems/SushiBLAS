/**************************************************************************/
/* syrk.hpp                                                               */
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
            * @brief SYCL dispatcher for SYRK using oneMKL.
            * 
            * Handles both single and batch SYRK.
            */
            template<typename T>
            [[nodiscard]] sycl::event syrk(sycl::queue& queue, Core::Layout layout,
                             oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                             int64_t n, int64_t k,
                             T alpha, const T* a, int64_t lda, int64_t str_a,
                             T beta, T* c, int64_t ldc, int64_t str_c,
                             int64_t batch_size, const std::vector<sycl::event>& deps)
            {
                if constexpr (std::is_same_v<T, sycl::half>)
                {
                    SB_THROW_IF(true, "oneMKL does not support sycl::half for SYRK. Use FORCE_REFERENCE or cast to float32.");
                    return sycl::event();
                }
                else
                {
                    if (layout == Core::Layout::ROW_MAJOR) 
                    {
                        if (batch_size > 1) 
                        {
                            SB_LOG_INFO("SYCL Batch SYRK [Row-Major]: {}x[{}x{}]", batch_size, n, n);
                            return oneapi::mkl::blas::row_major::syrk_batch(queue, uplo, trans, n, k, alpha, a, lda, str_a, beta, c, ldc, str_c, batch_size, deps);
                        } 
                        else 
                        {
                            SB_LOG_INFO("SYCL SYRK [Row-Major]: {}x{}", n, n);
                            return oneapi::mkl::blas::row_major::syrk(queue, uplo, trans, n, k, alpha, a, lda, beta, c, ldc, deps);
                        }
                    } 
                    else // COLUMN_MAJOR
                    {
                        if (batch_size > 1) 
                        {
                            SB_LOG_INFO("SYCL Batch SYRK [Col-Major]: {}x[{}x{}]", batch_size, n, n);
                            return oneapi::mkl::blas::column_major::syrk_batch(queue, uplo, trans, n, k, alpha, a, lda, str_a, beta, c, ldc, str_c, batch_size, deps);
                        } 
                        else 
                        {
                            SB_LOG_INFO("SYCL SYRK [Col-Major]: {}x{}", n, n);
                            return oneapi::mkl::blas::column_major::syrk(queue, uplo, trans, n, k, alpha, a, lda, beta, c, ldc, deps);
                        }
                    }
                }
            }
            
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
