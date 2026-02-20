/**************************************************************************/
/* syrk.cpp                                                               */
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
         * @brief Internal dispatcher for SYRK handling both Layout and Batching logic.
         */
        template<typename T>
        void syrk_dispatch(sycl::queue& queue, Core::Layout layout,
                           oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                           int64_t n, int64_t k,
                           T alpha, const T* a, int64_t lda, int64_t str_a,
                           T beta, T* c, int64_t ldc, int64_t str_c,
                           int64_t batch_size)
        {
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                if (batch_size > 1) 
                {
                    SB_LOG_INFO("MKL Batch SYRK [Row-Major]: {}x[{}x{}]", batch_size, n, n);
                    oneapi::mkl::blas::row_major::syrk_batch(queue, uplo, trans, n, k, alpha, a, lda, str_a, beta, c, ldc, str_c, batch_size, oneapi::mkl::blas::compute_mode::standard);
                } 
                else 
                {
                    SB_LOG_INFO("MKL SYRK [Row-Major]: {}x{}", n, n);
                    oneapi::mkl::blas::row_major::syrk(queue, uplo, trans, n, k, alpha, a, lda, beta, c, ldc, oneapi::mkl::blas::compute_mode::standard);
                }
            } 
            else // COLUMN_MAJOR
            {
                if (batch_size > 1) 
                {
                    SB_LOG_INFO("MKL Batch SYRK [Col-Major]: {}x[{}x{}]", batch_size, n, n);
                    oneapi::mkl::blas::column_major::syrk_batch(queue, uplo, trans, n, k, alpha, a, lda, str_a, beta, c, ldc, str_c, batch_size, oneapi::mkl::blas::compute_mode::standard);
                } 
                else 
                {
                    SB_LOG_INFO("MKL SYRK [Col-Major]: {}x{}", n, n);
                    oneapi::mkl::blas::column_major::syrk(queue, uplo, trans, n, k, alpha, a, lda, beta, c, ldc, oneapi::mkl::blas::compute_mode::standard);
                }
            }
        }
    }

    void Level3::syrk(const Tensor& A, Tensor& C, 
                      bool upper, bool transA, 
                      float alpha, float beta) 
    {
        // 1. Validation
        SB_THROW_IF(A.rank < 2 || C.rank < 2, "SYRK requires at least 2D tensors.");
        SB_THROW_IF(A.dtype != C.dtype, "Data type mismatch in SYRK operation.");

        int32_t rA = A.rank, rC = C.rank;
        
        // C must be strictly square n x n
        int64_t n = C.shape[rC - 2];
        SB_THROW_IF(C.shape[rC - 1] != n, "SYRK requires C to be a square matrix.");

        // Define inner k based on trans flag
        int64_t inner_n = transA ? A.shape[rA - 1] : A.shape[rA - 2];
        int64_t k       = transA ? A.shape[rA - 2] : A.shape[rA - 1];
        
        SB_THROW_IF(inner_n != n, "Dimension mismatch in SYRK: A does not match C dimension.");

        // Batch extraction
        int64_t batch_size = 1;
        for (int i = 0; i < rC - 2; ++i) batch_size *= C.shape[i];

        // Lda / Ldc
        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];
        int64_t ldc = (C.layout == Core::Layout::ROW_MAJOR) ? C.shape[rC - 1] : C.shape[rC - 2];

        // Strides
        int64_t str_a = (batch_size > 1) ? A.strides[rA - 3] : 0;
        int64_t str_c = (batch_size > 1) ? C.strides[rC - 3] : 0;

        auto& queue = engine_.get_context().get_queue();
        auto mkl_uplo = upper ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;
        auto mkl_trans = transA ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;

        // 2. Comprehensive Type Dispatching
        switch (A.dtype)
        {
            case Core::DataType::FLOAT32:
                syrk_dispatch<float>(queue, A.layout, mkl_uplo, mkl_trans, n, k, alpha, A.data_as<float>(), lda, str_a, beta, C.data_as<float>(), ldc, str_c, batch_size);
                break;
            case Core::DataType::FLOAT64:
                syrk_dispatch<double>(queue, A.layout, mkl_uplo, mkl_trans, n, k, static_cast<double>(alpha), A.data_as<double>(), lda, str_a, static_cast<double>(beta), C.data_as<double>(), ldc, str_c, batch_size);
                break;
            case Core::DataType::COMPLEX32:
                syrk_dispatch<std::complex<float>>(queue, A.layout, mkl_uplo, mkl_trans, n, k, std::complex<float>(alpha, 0.0f), A.data_as<std::complex<float>>(), lda, str_a, std::complex<float>(beta, 0.0f), C.data_as<std::complex<float>>(), ldc, str_c, batch_size);
                break;
            case Core::DataType::COMPLEX64:
                syrk_dispatch<std::complex<double>>(queue, A.layout, mkl_uplo, mkl_trans, n, k, std::complex<double>(alpha, 0.0), A.data_as<std::complex<double>>(), lda, str_a, std::complex<double>(beta, 0.0), C.data_as<std::complex<double>>(), ldc, str_c, batch_size);
                break;
            default:
                SB_THROW_IF(true, "Unsupported data type for SYRK operation.");
        }
    }
} // namespace SushiBLAS
