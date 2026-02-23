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

#include <vector>
#include <complex>
#include <oneapi/mkl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/ops/blas/level3.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /**
         * @brief Internal dispatcher for SYRK handling both Layout and Batching logic.
         */
        template<typename T>
        sycl::event syrk_dispatch(sycl::queue& queue, Core::Layout layout,
                           oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                           int64_t n, int64_t k,
                           T alpha, const T* a, int64_t lda, int64_t str_a,
                           T beta, T* c, int64_t ldc, int64_t str_c,
                           int64_t batch_size, const std::vector<sycl::event>& deps)
        {
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                if (batch_size > 1) 
                {
                    SB_LOG_INFO("MKL Batch SYRK [Row-Major]: {}x[{}x{}]", batch_size, n, n);
                    return oneapi::mkl::blas::row_major::syrk_batch(queue, uplo, trans, n, k, alpha, a, lda, str_a, beta, c, ldc, str_c, batch_size, oneapi::mkl::blas::compute_mode::standard, deps);
                } 
                else 
                {
                    SB_LOG_INFO("MKL SYRK [Row-Major]: {}x{}", n, n);
                    return oneapi::mkl::blas::row_major::syrk(queue, uplo, trans, n, k, alpha, a, lda, beta, c, ldc, oneapi::mkl::blas::compute_mode::standard, deps);
                }
            } 
            else // COLUMN_MAJOR
            {
                if (batch_size > 1) 
                {
                    SB_LOG_INFO("MKL Batch SYRK [Col-Major]: {}x[{}x{}]", batch_size, n, n);
                    return oneapi::mkl::blas::column_major::syrk_batch(queue, uplo, trans, n, k, alpha, a, lda, str_a, beta, c, ldc, str_c, batch_size, oneapi::mkl::blas::compute_mode::standard, deps);
                } 
                else 
                {
                    SB_LOG_INFO("MKL SYRK [Col-Major]: {}x{}", n, n);
                    return oneapi::mkl::blas::column_major::syrk(queue, uplo, trans, n, k, alpha, a, lda, beta, c, ldc, oneapi::mkl::blas::compute_mode::standard, deps);
                }
            }
        }
    }

    sycl::event Level3::syrk(const Tensor& A, Tensor& C, 
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

        auto mkl_uplo = upper ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;
        auto mkl_trans = transA ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;

        // Capture data safely for asynchronous execution
        auto layout = A.layout;
        void* read_A = A.storage ? A.storage->data_ptr : nullptr;
        void* write_C = C.storage ? C.storage->data_ptr : nullptr;

        std::vector<void*> reads = {};
        if (read_A) reads.push_back(read_A);
        std::vector<void*> writes = {};
        if (write_C) writes.push_back(write_C);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "mkl_syrk";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.syrk"_op;
        meta.set_param(0, alpha);
        meta.set_param(1, beta);
        meta.set_param(2, upper);
        meta.set_param(3, transA);

        // 2. Comprehensive Type Dispatching
        switch (A.dtype)
        {
            case Core::DataType::FLOAT32:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_uplo, mkl_trans, n, k, alpha, lda, str_a, beta, ldc, str_c, batch_size,
                     pA=A.data_as<float>(), pC=C.data_as<float>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        return syrk_dispatch<float>(q, layout, mkl_uplo, mkl_trans, n, k, alpha, pA, lda, str_a, beta, pC, ldc, str_c, batch_size, deps);
                    }
                );
                break;
            }
            case Core::DataType::FLOAT64:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_uplo, mkl_trans, n, k, alpha_d=static_cast<double>(alpha), lda, str_a, beta_d=static_cast<double>(beta), ldc, str_c, batch_size,
                     pA=A.data_as<double>(), pC=C.data_as<double>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        return syrk_dispatch<double>(q, layout, mkl_uplo, mkl_trans, n, k, alpha_d, pA, lda, str_a, beta_d, pC, ldc, str_c, batch_size, deps);
                    }
                );
                break;
            }
            case Core::DataType::COMPLEX32:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_uplo, mkl_trans, n, k, alpha_c=std::complex<float>(alpha, 0.0f), lda, str_a, beta_c=std::complex<float>(beta, 0.0f), ldc, str_c, batch_size,
                     pA=A.data_as<std::complex<float>>(), pC=C.data_as<std::complex<float>>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        return syrk_dispatch<std::complex<float>>(q, layout, mkl_uplo, mkl_trans, n, k, alpha_c, pA, lda, str_a, beta_c, pC, ldc, str_c, batch_size, deps);
                    }
                );
                break;
            }
            case Core::DataType::COMPLEX64:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_uplo, mkl_trans, n, k, alpha_c=std::complex<double>(alpha, 0.0), lda, str_a, beta_c=std::complex<double>(beta, 0.0), ldc, str_c, batch_size,
                     pA=A.data_as<std::complex<double>>(), pC=C.data_as<std::complex<double>>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        return syrk_dispatch<std::complex<double>>(q, layout, mkl_uplo, mkl_trans, n, k, alpha_c, pA, lda, str_a, beta_c, pC, ldc, str_c, batch_size, deps);
                    }
                );
                break;
            }
            default:
                SB_THROW_IF(true, "Unsupported data type for SYRK operation.");
        }
        return sycl::event();
    }
} // namespace SushiBLAS
