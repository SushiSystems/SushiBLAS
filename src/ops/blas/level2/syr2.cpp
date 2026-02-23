/**************************************************************************/
/* syr2.cpp                                                               */
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
#include <SushiBLAS/ops/blas/utils.hpp>
#include <SushiBLAS/ops/blas/level2.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        template<typename T>
        sycl::event syr2_dispatch(sycl::queue& queue, Core::Layout layout,
                           oneapi::mkl::uplo uplo,
                           int64_t n, T alpha, const T* x, int64_t incx,
                           const T* y, int64_t incy,
                           T* a, int64_t lda, const std::vector<sycl::event>& deps) 
        {
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                SB_LOG_INFO("MKL SYR2 [Row-Major]: {}x{}", n, n);
                return oneapi::mkl::blas::row_major::syr2(queue, uplo, n, alpha, x, incx, y, incy, a, lda, deps);
            } 
            else 
            {
                SB_LOG_INFO("MKL SYR2 [Col-Major]: {}x{}", n, n);
                return oneapi::mkl::blas::column_major::syr2(queue, uplo, n, alpha, x, incx, y, incy, a, lda, deps);
            }
        }
    }

    sycl::event Level2::syr2(const Tensor& x, const Tensor& y, Tensor& A, bool upper, float alpha) 
    {
        SB_THROW_IF(A.rank < 2, "SYR2 requires A to be at least a 2D matrix.");
        SB_THROW_IF(A.dtype != x.dtype || A.dtype != y.dtype, "Data type mismatch in SYR2.");

        int32_t rA = A.rank;
        int64_t n = A.shape[rA - 1]; // Symmetric matrix must be n x n
        SB_THROW_IF(A.shape[rA - 2] != n, "SYR2 requires A to be a square matrix.");

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];

        int64_t nx, incx;
        Internal::get_vec_params(x, nx, incx);
        
        int64_t ny, incy;
        Internal::get_vec_params(y, ny, incy);

        SB_THROW_IF(nx != n, "Dimension mismatch for vector x in SYR2.");
        SB_THROW_IF(ny != n, "Dimension mismatch for vector y in SYR2.");

        auto mkl_uplo = upper ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;
        auto layout = A.layout;

        void* read_x = x.storage ? x.storage->data_ptr : nullptr;
        void* read_y = y.storage ? y.storage->data_ptr : nullptr;
        void* write_A = A.storage ? A.storage->data_ptr : nullptr;

        std::vector<void*> reads = {};
        if (read_x) reads.push_back(read_x);
        if (read_y) reads.push_back(read_y);
        std::vector<void*> writes = {};
        if (write_A) writes.push_back(write_A);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "mkl_syr2";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.syr2"_op;
        meta.set_param(0, upper);
        meta.set_param(1, alpha);

        switch (A.dtype) 
        {
            case Core::DataType::FLOAT32:
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_uplo, n, alpha, lda, incx, incy, px=x.data_as<float>(), py=y.data_as<float>(), pA=A.data_as<float>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return syr2_dispatch<float>(q, layout, mkl_uplo, n, alpha, px, incx, py, incy, pA, lda, deps);
                    });
                break;
            case Core::DataType::FLOAT64:
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_uplo, n, alpha_d=static_cast<double>(alpha), lda, incx, incy, px=x.data_as<double>(), py=y.data_as<double>(), pA=A.data_as<double>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return syr2_dispatch<double>(q, layout, mkl_uplo, n, alpha_d, px, incx, py, incy, pA, lda, deps);
                    });
                break;
            case Core::DataType::COMPLEX32:
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_uplo, n, alpha_c=std::complex<float>(alpha, 0.0f), lda, incx, incy, px=x.data_as<std::complex<float>>(), py=y.data_as<std::complex<float>>(), pA=A.data_as<std::complex<float>>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return syr2_dispatch<std::complex<float>>(q, layout, mkl_uplo, n, alpha_c, px, incx, py, incy, pA, lda, deps);
                    });
                break;
            case Core::DataType::COMPLEX64:
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_uplo, n, alpha_c=std::complex<double>(alpha, 0.0), lda, incx, incy, px=x.data_as<std::complex<double>>(), py=y.data_as<std::complex<double>>(), pA=A.data_as<std::complex<double>>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return syr2_dispatch<std::complex<double>>(q, layout, mkl_uplo, n, alpha_c, px, incx, py, incy, pA, lda, deps);
                    });
                break;
            default:
                SB_THROW_IF(true, "Unsupported data type for SYR2.");
        }
        return sycl::event();
    }
}
