/**************************************************************************/
/* gemm.cpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                   	  */
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

#include <complex>
#include <oneapi/mkl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/ops/blas/level3.hpp>

namespace SushiBLAS 
{
    namespace
    {
        /**
         * @brief Internal dispatcher that handles both Layout and Batching logic.
         */
        template<typename T>
        void gemm_dispatch(sycl::queue& queue, Core::Layout layout,
                           oneapi::mkl::transpose transA, oneapi::mkl::transpose transB,
                           int64_t m, int64_t n, int64_t k,
                           T alpha, const T* a, int64_t lda, int64_t str_a,
                           const T* b, int64_t ldb, int64_t str_b,
                           T beta, T* c, int64_t ldc, int64_t str_c,
                           int64_t batch_size)
        {
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                if (batch_size > 1) 
                {
                    SB_LOG_INFO("MKL Batch GEMM [Row-Major]: {}x[{}x{}x{}]", batch_size, m, n, k);
                    oneapi::mkl::blas::row_major::gemm_batch(queue, transA, transB, m, n, k, alpha, a, lda, str_a, b, ldb, str_b, beta, c, ldc, str_c, batch_size, oneapi::mkl::blas::compute_mode::standard);
                } 
                else 
                {
                    SB_LOG_INFO("MKL GEMM [Row-Major]: {}x{}x{}", m, n, k);
                    oneapi::mkl::blas::row_major::gemm(queue, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, oneapi::mkl::blas::compute_mode::standard);
                }
            } 
            else // COLUMN_MAJOR
            {
                if (batch_size > 1) 
                {
                    SB_LOG_INFO("MKL Batch GEMM [Col-Major]: {}x[{}x{}x{}]", batch_size, m, n, k);
                    oneapi::mkl::blas::column_major::gemm_batch(queue, transA, transB, m, n, k, alpha, a, lda, str_a, b, ldb, str_b, beta, c, ldc, str_c, batch_size, oneapi::mkl::blas::compute_mode::standard);
                } 
                else 
                {
                    SB_LOG_INFO("MKL GEMM [Col-Major]: {}x{}x{}", m, n, k);
                    oneapi::mkl::blas::column_major::gemm(queue, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, oneapi::mkl::blas::compute_mode::standard);
                }
            }
        }
    } // namespace Anonymous

    void Level3::gemm(const Tensor& A, const Tensor& B, Tensor& C, 
                      bool transA, bool transB,
                      float alpha, float beta) 
    {
        // 1. Validation
        SB_THROW_IF(A.rank < 2 || B.rank < 2 || C.rank < 2, "GEMM requires at least 2D tensors.");
        SB_THROW_IF(A.dtype != B.dtype || A.dtype != C.dtype, "Data type mismatch in GEMM operation.");

        int32_t rA = A.rank, rB = B.rank, rC = C.rank;
        int64_t m = C.shape[rC - 2];
        int64_t n = C.shape[rC - 1];
        int64_t k = transA ? A.shape[rA - 2] : A.shape[rA - 1];

        // Batch & Stride calculation
        int64_t batch_size = 1;
        for (int i = 0; i < rC - 2; ++i) batch_size *= C.shape[i];

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];
        int64_t ldb = (B.layout == Core::Layout::ROW_MAJOR) ? B.shape[rB - 1] : B.shape[rB - 2];
        int64_t ldc = (C.layout == Core::Layout::ROW_MAJOR) ? C.shape[rC - 1] : C.shape[rC - 2];

        int64_t str_a = (batch_size > 1) ? A.strides[rA - 3] : 0;
        int64_t str_b = (batch_size > 1) ? B.strides[rB - 3] : 0;
        int64_t str_c = (batch_size > 1) ? C.strides[rC - 3] : 0;

        auto& queue = engine_.get_context().get_queue();
        auto mkl_transA = transA ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
        auto mkl_transB = transB ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;

        switch (A.dtype)
        {
            case Core::DataType::HALF:
                gemm_dispatch<sycl::half>(queue, A.layout, mkl_transA, mkl_transB, m, n, k, static_cast<sycl::half>(alpha), A.data_as<sycl::half>(), lda, str_a, B.data_as<sycl::half>(), ldb, str_b, static_cast<sycl::half>(beta), C.data_as<sycl::half>(), ldc, str_c, batch_size);
                break;
            case Core::DataType::FLOAT32:
                gemm_dispatch<float>(queue, A.layout, mkl_transA, mkl_transB, m, n, k, alpha, A.data_as<float>(), lda, str_a, B.data_as<float>(), ldb, str_b, beta, C.data_as<float>(), ldc, str_c, batch_size);
                break;
            case Core::DataType::FLOAT64:
                gemm_dispatch<double>(queue, A.layout, mkl_transA, mkl_transB, m, n, k, static_cast<double>(alpha), A.data_as<double>(), lda, str_a, B.data_as<double>(), ldb, str_b, static_cast<double>(beta), C.data_as<double>(), ldc, str_c, batch_size);
                break;
            case Core::DataType::COMPLEX32:
                gemm_dispatch<std::complex<float>>(queue, A.layout, mkl_transA, mkl_transB, m, n, k, std::complex<float>(alpha, 0.0f), A.data_as<std::complex<float>>(), lda, str_a, B.data_as<std::complex<float>>(), ldb, str_b, std::complex<float>(beta, 0.0f), C.data_as<std::complex<float>>(), ldc, str_c, batch_size);
                break;
            case Core::DataType::COMPLEX64:
                gemm_dispatch<std::complex<double>>(queue, A.layout, mkl_transA, mkl_transB, m, n, k, std::complex<double>(alpha, 0.0), A.data_as<std::complex<double>>(), lda, str_a, B.data_as<std::complex<double>>(), ldb, str_b, std::complex<double>(beta, 0.0), C.data_as<std::complex<double>>(), ldc, str_c, batch_size);
                break;
            default:
                SB_THROW_IF(true, "Unsupported data type for GEMM operation.");
        }
    }
} // namespace SushiBLAS
