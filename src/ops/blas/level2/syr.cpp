/**************************************************************************/
/* syr.cpp                                                                */
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
        sycl::event syr_dispatch(sycl::queue& queue, Core::Layout layout,
                          oneapi::mkl::uplo uplo,
                          int64_t n, T alpha, const T* x, int64_t incx,
                          T* a, int64_t lda, const std::vector<sycl::event>& deps) 
        {
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                SB_LOG_INFO("MKL SYR [Row-Major]: {}x{}", n, n);
                return oneapi::mkl::blas::row_major::syr(queue, uplo, n, alpha, x, incx, a, lda, deps);
            } 
            else 
            {
                SB_LOG_INFO("MKL SYR [Col-Major]: {}x{}", n, n);
                return oneapi::mkl::blas::column_major::syr(queue, uplo, n, alpha, x, incx, a, lda, deps);
            }
        }
    }

    sycl::event Level2::syr(const Tensor& x, Tensor& A, bool upper, float alpha) 
    {
        SB_THROW_IF(A.rank < 2, "SYR requires A to be at least a 2D matrix.");
        SB_THROW_IF(A.dtype != x.dtype, "Data type mismatch in SYR.");

        int32_t rA = A.rank;
        int64_t n = A.shape[rA - 1]; // Symmetric matrix must be n x n
        SB_THROW_IF(A.shape[rA - 2] != n, "SYR requires A to be a square matrix.");

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];

        int64_t nx, incx;
        Internal::get_vec_params(x, nx, incx);

        SB_THROW_IF(nx != n, "Dimension mismatch for vector x in SYR.");

        auto mkl_uplo = upper ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;
        auto layout = A.layout;

        void* read_x = x.storage ? x.storage->data_ptr : nullptr;
        void* write_A = A.storage ? A.storage->data_ptr : nullptr;

        std::vector<void*> reads = {};
        if (read_x) reads.push_back(read_x);
        std::vector<void*> writes = {};
        if (write_A) writes.push_back(write_A);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl2.syr";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl2.syr"_op;
        meta.set_param(0, upper);
        meta.set_param(1, alpha);

        // TODO: Implement multi-dimensional batch support for Level 2

        switch (A.dtype) 
        {
            // TODO: Add support for Core::DataType::HALF
            case Core::DataType::FLOAT32:
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_uplo, n, alpha, lda, incx, px=x.data_as<float>(), pA=A.data_as<float>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return syr_dispatch<float>(q, layout, mkl_uplo, n, alpha, px, incx, pA, lda, deps);
                    });
                break;
            case Core::DataType::FLOAT64:
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_uplo, n, alpha_d=static_cast<double>(alpha), lda, incx, px=x.data_as<double>(), pA=A.data_as<double>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return syr_dispatch<double>(q, layout, mkl_uplo, n, alpha_d, px, incx, pA, lda, deps);
                    });
                break;
            case Core::DataType::COMPLEX32:
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_uplo, n, alpha_c=std::complex<float>(alpha, 0.0f), lda, incx, px=x.data_as<std::complex<float>>(), pA=A.data_as<std::complex<float>>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return syr_dispatch<std::complex<float>>(q, layout, mkl_uplo, n, alpha_c, px, incx, pA, lda, deps);
                    });
                break;
            case Core::DataType::COMPLEX64:
                engine_.get_graph().add_task(meta, reads, writes,
                    [layout, mkl_uplo, n, alpha_c=std::complex<double>(alpha, 0.0), lda, incx, px=x.data_as<std::complex<double>>(), pA=A.data_as<std::complex<double>>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return syr_dispatch<std::complex<double>>(q, layout, mkl_uplo, n, alpha_c, px, incx, pA, lda, deps);
                    });
                break;
            default:
                SB_THROW_IF(true, "Unsupported data type for SYR.");
        }
        return sycl::event();
    }
} // namespace SushiBLAS
