/**************************************************************************/
/* ger.cpp                                                                */
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
        sycl::event ger_dispatch(sycl::queue& queue, Core::Layout layout,
                          int64_t m, int64_t n,
                          T alpha, const T* x, int64_t incx,
                          const T* y, int64_t incy,
                          T* a, int64_t lda, const std::vector<sycl::event>& deps) 
        {
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                SB_LOG_INFO("MKL GER [Row-Major]: {}x{}", m, n);
                return oneapi::mkl::blas::row_major::geru(queue, m, n, alpha, x, incx, y, incy, a, lda, deps);
            } 
            else 
            {
                SB_LOG_INFO("MKL GER [Col-Major]: {}x{}", m, n);
                return oneapi::mkl::blas::column_major::geru(queue, m, n, alpha, x, incx, y, incy, a, lda, deps);
            }
        }
        
        template<>
        sycl::event ger_dispatch<float>(sycl::queue& queue, Core::Layout layout, int64_t m, int64_t n, float alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* a, int64_t lda, const std::vector<sycl::event>& deps) 
        {
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                SB_LOG_INFO("MKL GER [Row-Major]: {}x{}", m, n);
                return oneapi::mkl::blas::row_major::ger(queue, m, n, alpha, x, incx, y, incy, a, lda, deps);
            } else {
                SB_LOG_INFO("MKL GER [Col-Major]: {}x{}", m, n);
                return oneapi::mkl::blas::column_major::ger(queue, m, n, alpha, x, incx, y, incy, a, lda, deps);
            }
        }
        
        template<>
        sycl::event ger_dispatch<double>(sycl::queue& queue, Core::Layout layout, int64_t m, int64_t n, double alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* a, int64_t lda, const std::vector<sycl::event>& deps) 
        {
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                SB_LOG_INFO("MKL GER [Row-Major]: {}x{}", m, n);
                return oneapi::mkl::blas::row_major::ger(queue, m, n, alpha, x, incx, y, incy, a, lda, deps);
            } else {
                SB_LOG_INFO("MKL GER [Col-Major]: {}x{}", m, n);
                return oneapi::mkl::blas::column_major::ger(queue, m, n, alpha, x, incx, y, incy, a, lda, deps);
            }
        }
    }

    sycl::event Level2::ger(const Tensor& x, const Tensor& y, Tensor& A, float alpha) 
    {
        SB_THROW_IF(A.rank < 2, "GER requires A to be at least a 2D matrix.");
        SB_THROW_IF(A.dtype != x.dtype || A.dtype != y.dtype, "Data type mismatch in GER.");

        int32_t rA = A.rank;
        int64_t m = A.shape[rA - 2];
        int64_t n = A.shape[rA - 1];

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];

        int64_t nx, incx;
        Internal::get_vec_params(x, nx, incx);
        
        int64_t ny, incy;
        Internal::get_vec_params(y, ny, incy);

        SB_THROW_IF(nx != m, "Dimension mismatch for vector x in GER.");
        SB_THROW_IF(ny != n, "Dimension mismatch for vector y in GER.");

        void* read_x = x.storage ? x.storage->data_ptr : nullptr;
        void* read_y = y.storage ? y.storage->data_ptr : nullptr;
        void* write_A = A.storage ? A.storage->data_ptr : nullptr;

        std::vector<void*> reads = {};
        if (read_x) reads.push_back(read_x);
        if (read_y) reads.push_back(read_y);
        std::vector<void*> writes = {};
        if (write_A) writes.push_back(write_A);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl2.ger";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl2.ger"_op;
        meta.set_param(0, alpha);

        auto layout = A.layout;

        // TODO: Implement multi-dimensional batch support for Level 2

        switch (A.dtype) 
        {
            // TODO: Add support for Core::DataType::HALF
            case Core::DataType::FLOAT32:
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, m, n, alpha, incx, incy, lda, px=x.data_as<float>(), py=y.data_as<float>(), pA=A.data_as<float>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return ger_dispatch<float>(q, layout, m, n, alpha, px, incx, py, incy, pA, lda, deps);
                    });
                break;
            case Core::DataType::FLOAT64:
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, m, n, alpha_d=static_cast<double>(alpha), incx, incy, lda, px=x.data_as<double>(), py=y.data_as<double>(), pA=A.data_as<double>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return ger_dispatch<double>(q, layout, m, n, alpha_d, px, incx, py, incy, pA, lda, deps);
                    });
                break;
            case Core::DataType::COMPLEX32:
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, m, n, alpha_c=std::complex<float>(alpha, 0.0f), incx, incy, lda, px=x.data_as<std::complex<float>>(), py=y.data_as<std::complex<float>>(), pA=A.data_as<std::complex<float>>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return ger_dispatch<std::complex<float>>(q, layout, m, n, alpha_c, px, incx, py, incy, pA, lda, deps);
                    });
                break;
            case Core::DataType::COMPLEX64:
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, m, n, alpha_c=std::complex<double>(alpha, 0.0), incx, incy, lda, px=x.data_as<std::complex<double>>(), py=y.data_as<std::complex<double>>(), pA=A.data_as<std::complex<double>>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return ger_dispatch<std::complex<double>>(q, layout, m, n, alpha_c, px, incx, py, incy, pA, lda, deps);
                    });
                break;
            default:
                SB_THROW_IF(true, "Unsupported data type for GER.");
        }
        return sycl::event();
    }
}
