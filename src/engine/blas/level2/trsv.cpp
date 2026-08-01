/**************************************************************************/
/* trsv.cpp                                                               */
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

#include <complex>
#include <oneapi/mkl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/blas/utils.hpp>
#include <SushiBLAS/engine/blas/level2.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        template<typename T>
        sycl::event trsv_dispatch(sycl::queue& queue, Core::Layout layout,
                           oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, oneapi::mkl::diag diag,
                           int64_t n, const T* a, int64_t lda,
                           T* x, int64_t incx, const std::vector<sycl::event>& deps) 
        {
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                SB_LOG_INFO("MKL TRSV [Row-Major]: {}x{}", n, n);
                return oneapi::mkl::blas::row_major::trsv(queue, uplo, trans, diag, n, a, lda, x, incx, deps);
            } 
            else 
            {
                SB_LOG_INFO("MKL TRSV [Col-Major]: {}x{}", n, n);
                return oneapi::mkl::blas::column_major::trsv(queue, uplo, trans, diag, n, a, lda, x, incx, deps);
            }
        }
    }

    sycl::event Level2::trsv(const Tensor& A, Tensor& b, 
                             bool upper, bool transA, bool unit_diag) 
    {
        SB_THROW_IF(A.rank < 2, "TRSV requires A to be at least a 2D matrix.");
        SB_THROW_IF(A.dtype != b.dtype, "Data type mismatch in TRSV.");

        int32_t rA = A.rank;
        int64_t n = A.shape[rA - 1]; // Triangular matrix must be n x n
        SB_THROW_IF(A.shape[rA - 2] != n, "TRSV requires A to be a square matrix.");

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];

        int64_t nx, incx;
        Internal::get_vec_params(b, nx, incx);

        SB_THROW_IF(nx != n, "Dimension mismatch for vector b in TRSV.");

        auto mkl_uplo = upper ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;
        auto mkl_trans = transA ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
        auto mkl_diag = unit_diag ? oneapi::mkl::diag::unit : oneapi::mkl::diag::nonunit;
        auto layout = A.layout;

        void* read_A = A.storage ? A.storage->data_ptr : nullptr;
        void* write_b = b.storage ? b.storage->data_ptr : nullptr; // TRSV overwrites b

        std::vector<void*> reads = {};
        if (read_A) reads.push_back(read_A);
        std::vector<void*> writes = {};
        if (write_b) writes.push_back(write_b);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl2.trsv";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl2.trsv"_op;
        meta.set_param(0, upper);
        meta.set_param(1, transA);
        meta.set_param(2, unit_diag);

        // TODO: Implement multi-dimensional batch support for Level 2

        switch (A.dtype) 
        {
            // TODO: Add support for Core::DataType::HALF
            case Core::DataType::FLOAT32:
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_uplo, mkl_trans, mkl_diag, n, lda, incx, pA=A.data_as<float>(), pb=b.data_as<float>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return trsv_dispatch<float>(q, layout, mkl_uplo, mkl_trans, mkl_diag, n, pA, lda, pb, incx, deps);
                    });
                break;
            case Core::DataType::FLOAT64:
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_uplo, mkl_trans, mkl_diag, n, lda, incx, pA=A.data_as<double>(), pb=b.data_as<double>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return trsv_dispatch<double>(q, layout, mkl_uplo, mkl_trans, mkl_diag, n, pA, lda, pb, incx, deps);
                    });
                break;
            case Core::DataType::COMPLEX32:
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_uplo, mkl_trans, mkl_diag, n, lda, incx, pA=A.data_as<std::complex<float>>(), pb=b.data_as<std::complex<float>>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return trsv_dispatch<std::complex<float>>(q, layout, mkl_uplo, mkl_trans, mkl_diag, n, pA, lda, pb, incx, deps);
                    });
                break;
            case Core::DataType::COMPLEX64:
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_uplo, mkl_trans, mkl_diag, n, lda, incx, pA=A.data_as<std::complex<double>>(), pb=b.data_as<std::complex<double>>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return trsv_dispatch<std::complex<double>>(q, layout, mkl_uplo, mkl_trans, mkl_diag, n, pA, lda, pb, incx, deps);
                    });
                break;
            default:
                SB_THROW_IF(true, "Unsupported data type for TRSV.");
        }
        return sycl::event();
    }
}
