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

#include <vector>
#include <complex>
#include <kernels/blas.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/blas/level3.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event Level3::gemm(const Tensor& A, const Tensor& B, Tensor& C, 
                            bool transA, bool transB,
                            double alpha, double beta) 
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

        auto mkl_transA = transA ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
        auto mkl_transB = transB ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;

        // Capture data safely for asynchronous execution
        auto layout = A.layout;
        void* read_A = A.storage ? A.storage->data_ptr : nullptr;
        void* read_B = B.storage ? B.storage->data_ptr : nullptr;
        void* write_C = C.storage ? C.storage->data_ptr : nullptr;

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_A) reads.push_back(read_A);
        if (read_B) reads.push_back(read_B);
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_C) writes.push_back(write_C);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl3.gemm";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl3.gemm"_op;
        meta.set_param(0, alpha);
        meta.set_param(1, beta);
        meta.set_param(2, transA);
        meta.set_param(3, transB);

        auto mode = engine_.get_mode();

        switch (A.dtype)
        {
            case Core::DataType::HALF:
            {
                auto pA = A.data_as<sycl::half>();
                auto pB = B.data_as<sycl::half>();
                auto pC = C.data_as<sycl::half>();
                auto alpha_h = static_cast<sycl::half>(alpha);
                auto beta_h = static_cast<sycl::half>(beta);

                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_transA, mkl_transB, m, n, k, alpha_h, lda, str_a, ldb, str_b, beta_h, ldc, str_c, batch_size, 
                     pA, pB, pC, mode, transA, transB]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        if (mode == ExecutionMode::FORCE_REFERENCE)
                        {
                            sycl::event::wait(deps);
                            Kernels::Reference::gemm<sycl::half>(layout, transA, transB, m, n, k, alpha_h, pA, lda, str_a, pB, ldb, str_b, beta_h, pC, ldc, str_c, batch_size);
                            return sycl::event();
                        }

                        auto ev = Kernels::SYCL::gemm<sycl::half>(q, layout, mkl_transA, mkl_transB, m, n, k, alpha_h, pA, lda, str_a, pB, ldb, str_b, beta_h, pC, ldc, str_c, batch_size, deps);

                        if (mode == ExecutionMode::VERIFY)
                        {
                            ev.wait();
                            Kernels::Reference::gemm<sycl::half>(layout, transA, transB, m, n, k, alpha_h, pA, lda, str_a, pB, ldb, str_b, beta_h, pC, ldc, str_c, batch_size);
                        }
                        return ev;
                    }
                );
                break;
            }
            case Core::DataType::FLOAT32:
            {
                auto pA = A.data_as<float>();
                auto pB = B.data_as<float>();
                auto pC = C.data_as<float>();
                auto alpha_f = static_cast<float>(alpha);
                auto beta_f = static_cast<float>(beta);

                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_transA, mkl_transB, m, n, k, alpha_f, lda, str_a, ldb, str_b, beta_f, ldc, str_c, batch_size, 
                     pA, pB, pC, mode, transA, transB]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        if (mode == ExecutionMode::FORCE_REFERENCE)
                        {
                            sycl::event::wait(deps);
                            Kernels::Reference::gemm<float>(layout, transA, transB, m, n, k, alpha_f, pA, lda, str_a, pB, ldb, str_b, beta_f, pC, ldc, str_c, batch_size);
                            return sycl::event();
                        }

                        auto ev = Kernels::SYCL::gemm<float>(q, layout, mkl_transA, mkl_transB, m, n, k, alpha_f, pA, lda, str_a, pB, ldb, str_b, beta_f, pC, ldc, str_c, batch_size, deps);

                        if (mode == ExecutionMode::VERIFY)
                        {
                            ev.wait();
                            Kernels::Reference::gemm<float>(layout, transA, transB, m, n, k, alpha_f, pA, lda, str_a, pB, ldb, str_b, beta_f, pC, ldc, str_c, batch_size);
                        }
                        return ev;
                    }
                );
                break;
            }
            case Core::DataType::FLOAT64:
            {
                auto pA = A.data_as<double>();
                auto pB = B.data_as<double>();
                auto pC = C.data_as<double>();
                auto alpha_d = static_cast<double>(alpha);
                auto beta_d = static_cast<double>(beta);

                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_transA, mkl_transB, m, n, k, alpha_d, lda, str_a, ldb, str_b, beta_d, ldc, str_c, batch_size, 
                     pA, pB, pC, mode, transA, transB]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        if (mode == ExecutionMode::FORCE_REFERENCE)
                        {
                            sycl::event::wait(deps);
                            Kernels::Reference::gemm<double>(layout, transA, transB, m, n, k, alpha_d, pA, lda, str_a, pB, ldb, str_b, beta_d, pC, ldc, str_c, batch_size);
                            return sycl::event();
                        }

                        auto ev = Kernels::SYCL::gemm<double>(q, layout, mkl_transA, mkl_transB, m, n, k, alpha_d, pA, lda, str_a, pB, ldb, str_b, beta_d, pC, ldc, str_c, batch_size, deps);

                        if (mode == ExecutionMode::VERIFY)
                        {
                            ev.wait();
                            Kernels::Reference::gemm<double>(layout, transA, transB, m, n, k, alpha_d, pA, lda, str_a, pB, ldb, str_b, beta_d, pC, ldc, str_c, batch_size);
                        }
                        return ev;
                    }
                );
                break;
            }
            case Core::DataType::COMPLEX32:
            {
                auto pA = A.data_as<std::complex<float>>();
                auto pB = B.data_as<std::complex<float>>();
                auto pC = C.data_as<std::complex<float>>();
                auto alpha_c = std::complex<float>(static_cast<float>(alpha), 0.0f);
                auto beta_c = std::complex<float>(static_cast<float>(beta), 0.0f);

                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_transA, mkl_transB, m, n, k, alpha_c, lda, str_a, ldb, str_b, beta_c, ldc, str_c, batch_size, 
                     pA, pB, pC, mode, transA, transB]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        if (mode == ExecutionMode::FORCE_REFERENCE)
                        {
                            sycl::event::wait(deps);
                            Kernels::Reference::gemm<std::complex<float>>(layout, transA, transB, m, n, k, alpha_c, pA, lda, str_a, pB, ldb, str_b, beta_c, pC, ldc, str_c, batch_size);
                            return sycl::event();
                        }

                        auto ev = Kernels::SYCL::gemm<std::complex<float>>(q, layout, mkl_transA, mkl_transB, m, n, k, alpha_c, pA, lda, str_a, pB, ldb, str_b, beta_c, pC, ldc, str_c, batch_size, deps);

                        if (mode == ExecutionMode::VERIFY)
                        {
                            ev.wait();
                            Kernels::Reference::gemm<std::complex<float>>(layout, transA, transB, m, n, k, alpha_c, pA, lda, str_a, pB, ldb, str_b, beta_c, pC, ldc, str_c, batch_size);
                        }
                        return ev;
                    }
                );
                break;
            }
            case Core::DataType::COMPLEX64:
            {
                auto pA = A.data_as<std::complex<double>>();
                auto pB = B.data_as<std::complex<double>>();
                auto pC = C.data_as<std::complex<double>>();
                auto alpha_c = std::complex<double>(alpha, 0.0);
                auto beta_c = std::complex<double>(beta, 0.0);

                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_transA, mkl_transB, m, n, k, alpha_c, lda, str_a, ldb, str_b, beta_c, ldc, str_c, batch_size, 
                     pA, pB, pC, mode, transA, transB]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        if (mode == ExecutionMode::FORCE_REFERENCE)
                        {
                            sycl::event::wait(deps);
                            Kernels::Reference::gemm<std::complex<double>>(layout, transA, transB, m, n, k, alpha_c, pA, lda, str_a, pB, ldb, str_b, beta_c, pC, ldc, str_c, batch_size);
                            return sycl::event();
                        }

                        auto ev = Kernels::SYCL::gemm<std::complex<double>>(q, layout, mkl_transA, mkl_transB, m, n, k, alpha_c, pA, lda, str_a, pB, ldb, str_b, beta_c, pC, ldc, str_c, batch_size, deps);

                        if (mode == ExecutionMode::VERIFY)
                        {
                            ev.wait();
                            Kernels::Reference::gemm<std::complex<double>>(layout, transA, transB, m, n, k, alpha_c, pA, lda, str_a, pB, ldb, str_b, beta_c, pC, ldc, str_c, batch_size);
                        }
                        return ev;
                    }
                );
                break;
            }
            default:
                SB_THROW_IF(true, "Unsupported data type for GEMM operation.");
        }
        return sycl::event(); 
    }
} // namespace SushiBLAS
