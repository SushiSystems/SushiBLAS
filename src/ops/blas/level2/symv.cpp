/**************************************************************************/
/* symv.cpp                                                               */
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
#include <SushiBLAS/ops/blas/utils.hpp>
#include <SushiBLAS/ops/blas/level2.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        template<typename T>
        sycl::event symv_dispatch(sycl::queue& queue, Core::Layout layout,
                           oneapi::mkl::uplo uplo,
                           int64_t n, T alpha, const T* a, int64_t lda,
                           const T* x, int64_t incx,
                           T beta, T* y, int64_t incy, const std::vector<sycl::event>& deps) 
        {
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                SB_LOG_INFO("MKL SYMV [Row-Major]: {}x{}", n, n);
                return oneapi::mkl::blas::row_major::symv(queue, uplo, n, alpha, a, lda, x, incx, beta, y, incy, deps);
            } 
            else 
            {
                SB_LOG_INFO("MKL SYMV [Col-Major]: {}x{}", n, n);
                return oneapi::mkl::blas::column_major::symv(queue, uplo, n, alpha, a, lda, x, incx, beta, y, incy, deps);
            }
        }
    }

    sycl::event Level2::symv(const Tensor& A, const Tensor& x, Tensor& y, 
                             bool upper, float alpha, float beta) 
    {
        SB_THROW_IF(A.rank < 2, "SYMV requires A to be at least a 2D matrix.");
        SB_THROW_IF(A.dtype != x.dtype || A.dtype != y.dtype, "Data type mismatch in SYMV.");

        int32_t rA = A.rank;
        int64_t n = A.shape[rA - 1]; // Symmetric matrix must be n x n
        SB_THROW_IF(A.shape[rA - 2] != n, "SYMV requires A to be a square matrix.");

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];

        int64_t nx, incx;
        Internal::get_vec_params(x, nx, incx);
        
        int64_t ny, incy;
        Internal::get_vec_params(y, ny, incy);

        SB_THROW_IF(nx != n, "Dimension mismatch for vector x in SYMV.");
        SB_THROW_IF(ny != n, "Dimension mismatch for vector y in SYMV.");

        auto mkl_uplo = upper ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;
        auto layout = A.layout;

        void* read_A = A.storage ? A.storage->data_ptr : nullptr;
        void* read_x = x.storage ? x.storage->data_ptr : nullptr;
        void* write_y = y.storage ? y.storage->data_ptr : nullptr;

        std::vector<void*> reads = {};
        if (read_A) reads.push_back(read_A);
        if (read_x) reads.push_back(read_x);
        std::vector<void*> writes = {};
        if (write_y) writes.push_back(write_y);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl2.symv";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl2.symv"_op;
        meta.set_param(0, upper);
        meta.set_param(1, alpha);
        meta.set_param(2, beta);

        // TODO: Implement multi-dimensional batch support for Level 2

        switch (A.dtype) 
        {
            // TODO: Add support for Core::DataType::HALF
            case Core::DataType::FLOAT32:
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_uplo, n, alpha, lda, incx, beta, incy, pA=A.data_as<float>(), px=x.data_as<float>(), py=y.data_as<float>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return symv_dispatch<float>(q, layout, mkl_uplo, n, alpha, pA, lda, px, incx, beta, py, incy, deps);
                    });
                break;
            case Core::DataType::FLOAT64:
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_uplo, n, alpha_d=static_cast<double>(alpha), lda, incx, beta_d=static_cast<double>(beta), incy, pA=A.data_as<double>(), px=x.data_as<double>(), py=y.data_as<double>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return symv_dispatch<double>(q, layout, mkl_uplo, n, alpha_d, pA, lda, px, incx, beta_d, py, incy, deps);
                    });
                break;
            case Core::DataType::COMPLEX32:
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_uplo, n, alpha_c=std::complex<float>(alpha, 0.0f), lda, incx, beta_c=std::complex<float>(beta, 0.0f), incy, pA=A.data_as<std::complex<float>>(), px=x.data_as<std::complex<float>>(), py=y.data_as<std::complex<float>>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return symv_dispatch<std::complex<float>>(q, layout, mkl_uplo, n, alpha_c, pA, lda, px, incx, beta_c, py, incy, deps);
                    });
                break;
            case Core::DataType::COMPLEX64:
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_uplo, n, alpha_c=std::complex<double>(alpha, 0.0), lda, incx, beta_c=std::complex<double>(beta, 0.0), incy, pA=A.data_as<std::complex<double>>(), px=x.data_as<std::complex<double>>(), py=y.data_as<std::complex<double>>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return symv_dispatch<std::complex<double>>(q, layout, mkl_uplo, n, alpha_c, pA, lda, px, incx, beta_c, py, incy, deps);
                    });
                break;
            default:
                SB_THROW_IF(true, "Unsupported data type for SYMV.");
        }
        return sycl::event();
    }
}
