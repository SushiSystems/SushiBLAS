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
