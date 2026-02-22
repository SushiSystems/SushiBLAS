/**************************************************************************/
/* trsm.cpp                                                               */
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
#include "SushiRuntime/graph/task_types.hpp" // Added this include based on the instruction's context
#include <vector> // Added this include based on the instruction's context

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /**
         * @brief Internal dispatcher for TRSM handling Layout and Batching logic.
         */
        template<typename T>
        sycl::event trsm_dispatch(sycl::queue& queue, Core::Layout layout,
                           oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                           oneapi::mkl::transpose trans, oneapi::mkl::diag diag,
                           int64_t m, int64_t n, 
                           T alpha, const T* a, int64_t lda, int64_t str_a,
                           T* b, int64_t ldb, int64_t str_b,
                           int64_t batch_size, const std::vector<sycl::event>& deps)
        {
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                if (batch_size > 1) 
                {
                    SB_LOG_INFO("MKL Batch TRSM [Row-Major]: {}x[{}x{}]", batch_size, m, n);
                    return oneapi::mkl::blas::row_major::trsm_batch(queue, side, uplo, trans, diag, m, n, alpha, a, lda, str_a, b, ldb, str_b, batch_size, oneapi::mkl::blas::compute_mode::standard, deps);
                } 
                else 
                {
                    SB_LOG_INFO("MKL TRSM [Row-Major]: {}x{}", m, n);
                    return oneapi::mkl::blas::row_major::trsm(queue, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, oneapi::mkl::blas::compute_mode::standard, deps);
                }
            } 
            else // COLUMN_MAJOR
            {
                if (batch_size > 1) 
                {
                    SB_LOG_INFO("MKL Batch TRSM [Col-Major]: {}x[{}x{}]", batch_size, m, n);
                    return oneapi::mkl::blas::column_major::trsm_batch(queue, side, uplo, trans, diag, m, n, alpha, a, lda, str_a, b, ldb, str_b, batch_size, oneapi::mkl::blas::compute_mode::standard, deps);
                } 
                else 
                {
                    SB_LOG_INFO("MKL TRSM [Col-Major]: {}x{}", m, n);
                    return oneapi::mkl::blas::column_major::trsm(queue, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, oneapi::mkl::blas::compute_mode::standard, deps);
                }
            }
        }
    }

    sycl::event Level3::trsm(const Tensor& A, Tensor& B, 
                      bool left_side, bool upper, 
                      bool transA, bool unit_diag, 
                      float alpha) 
    {
        // 1. Validation
        SB_THROW_IF(A.rank < 2 || B.rank < 2, "TRSM requires at least 2D tensors.");
        SB_THROW_IF(A.dtype != B.dtype, "Data type mismatch in TRSM operation.");

        int32_t rA = A.rank, rB = B.rank;
        int64_t m = B.shape[rB - 2];
        int64_t n = B.shape[rB - 1];

        // A is expected to be a square matrix
        int64_t a_rows = A.shape[rA - 2];
        int64_t a_cols = A.shape[rA - 1];
        SB_THROW_IF(a_rows != a_cols, "TRSM requires matrix A to be square.");

        if (left_side) 
        {
            SB_THROW_IF(a_rows != m, "Left-sided TRSM: dimensions of A must match rows of B.");
        } 
        else 
        {
            SB_THROW_IF(a_rows != n, "Right-sided TRSM: dimensions of A must match cols of B.");
        }

        // Batch Extraction based on B
        int64_t batch_size = 1;
        for (int i = 0; i < rB - 2; ++i) batch_size *= B.shape[i];

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];
        int64_t ldb = (B.layout == Core::Layout::ROW_MAJOR) ? B.shape[rB - 1] : B.shape[rB - 2];

        int64_t str_a = (batch_size > 1) ? A.strides[rA - 3] : 0;
        int64_t str_b = (batch_size > 1) ? B.strides[rB - 3] : 0;

        auto mkl_side   = left_side ? oneapi::mkl::side::left : oneapi::mkl::side::right;
        auto mkl_uplo   = upper ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;
        auto mkl_trans  = transA ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
        auto mkl_diag   = unit_diag ? oneapi::mkl::diag::unit : oneapi::mkl::diag::nonunit;

        // Capture data safely for asynchronous execution
        auto layout = A.layout;
        void* read_A = A.storage ? A.storage->data_ptr : nullptr;
        void* write_B = B.storage ? B.storage->data_ptr : nullptr;

        std::vector<void*> reads = {};
        if (read_A) reads.push_back(read_A);
        std::vector<void*> writes = {};
        if (write_B) writes.push_back(write_B);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "mkl_trsm";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.trsm"_op;
        meta.set_param(0, alpha);
        meta.set_param(1, left_side);
        meta.set_param(2, upper);
        meta.set_param(3, transA);
        meta.set_param(4, unit_diag);

        // 2. Comprehensive Type Dispatching
        switch (A.dtype)
        {
            case Core::DataType::FLOAT32:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_side, mkl_uplo, mkl_trans, mkl_diag, m, n, alpha, lda, str_a, ldb, str_b, batch_size,
                     pA=A.data_as<float>(), pB=B.data_as<float>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        return trsm_dispatch<float>(q, layout, mkl_side, mkl_uplo, mkl_trans, mkl_diag, m, n, alpha, pA, lda, str_a, pB, ldb, str_b, batch_size, deps);
                    }
                );
                break;
            }
            case Core::DataType::FLOAT64:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_side, mkl_uplo, mkl_trans, mkl_diag, m, n, alpha_d=static_cast<double>(alpha), lda, str_a, ldb, str_b, batch_size,
                     pA=A.data_as<double>(), pB=B.data_as<double>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        return trsm_dispatch<double>(q, layout, mkl_side, mkl_uplo, mkl_trans, mkl_diag, m, n, alpha_d, pA, lda, str_a, pB, ldb, str_b, batch_size, deps);
                    }
                );
                break;
            }
            case Core::DataType::COMPLEX32:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_side, mkl_uplo, mkl_trans, mkl_diag, m, n, alpha_c=std::complex<float>(alpha, 0.0f), lda, str_a, ldb, str_b, batch_size,
                     pA=A.data_as<std::complex<float>>(), pB=B.data_as<std::complex<float>>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        return trsm_dispatch<std::complex<float>>(q, layout, mkl_side, mkl_uplo, mkl_trans, mkl_diag, m, n, alpha_c, pA, lda, str_a, pB, ldb, str_b, batch_size, deps);
                    }
                );
                break;
            }
            case Core::DataType::COMPLEX64:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_side, mkl_uplo, mkl_trans, mkl_diag, m, n, alpha_c=std::complex<double>(alpha, 0.0), lda, str_a, ldb, str_b, batch_size,
                     pA=A.data_as<std::complex<double>>(), pB=B.data_as<std::complex<double>>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        return trsm_dispatch<std::complex<double>>(q, layout, mkl_side, mkl_uplo, mkl_trans, mkl_diag, m, n, alpha_c, pA, lda, str_a, pB, ldb, str_b, batch_size, deps);
                    }
                );
                break;
            }
            default:
                SB_THROW_IF(true, "Unsupported data type for TRSM operation.");
        }
        return sycl::event();
    }
} // namespace SushiBLAS
