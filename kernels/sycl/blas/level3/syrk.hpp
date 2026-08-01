/**************************************************************************/
/* syrk.hpp                                                               */
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
