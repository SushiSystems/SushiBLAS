/**************************************************************************/
/* gemv.cpp                                                               */
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
#include <SushiBLAS/engine/blas/utils.hpp>
#include <SushiBLAS/engine/blas/level2.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        template<typename T>
        sycl::event gemv_dispatch(sycl::queue& queue, Core::Layout layout,
                           oneapi::mkl::transpose transA,
                           int64_t m, int64_t n,
                           T alpha, const T* a, int64_t lda,
                           const T* x, int64_t incx,
                           T beta, T* y, int64_t incy, const std::vector<sycl::event>& deps) 
        {
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                SB_LOG_INFO("MKL GEMV [Row-Major]: {}x{}", m, n);
                return oneapi::mkl::blas::row_major::gemv(queue, transA, m, n, alpha, a, lda, x, incx, beta, y, incy, deps);
            } 
            else 
            {
                SB_LOG_INFO("MKL GEMV [Col-Major]: {}x{}", m, n);
                return oneapi::mkl::blas::column_major::gemv(queue, transA, m, n, alpha, a, lda, x, incx, beta, y, incy, deps);
            }
        }
    }

    sycl::event Level2::gemv(const Tensor& A, const Tensor& x, Tensor& y,
                             bool transA, float alpha, float beta) 
    {
        SB_THROW_IF(A.rank < 2, "GEMV requires A to be at least a 2D matrix.");
        SB_THROW_IF(A.dtype != x.dtype || A.dtype != y.dtype, "Data type mismatch in GEMV.");

        int32_t rA = A.rank;
        int64_t m = A.shape[rA - 2];
        int64_t n = A.shape[rA - 1];

        // TODO: Implement multi-dimensional batch support for Level 2 GEMV

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];

        int64_t nx, incx;
        Internal::get_vec_params(x, nx, incx);
        
        int64_t ny, incy;
        Internal::get_vec_params(y, ny, incy);

        int64_t expected_nx = transA ? m : n;
        int64_t expected_ny = transA ? n : m;

        SB_THROW_IF(nx != expected_nx, "Dimension mismatch for vector x in GEMV.");
        SB_THROW_IF(ny != expected_ny, "Dimension mismatch for vector y in GEMV.");

        auto mkl_transA = transA ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;

        void* read_A = A.storage ? A.storage->data_ptr : nullptr;
        void* read_x = x.storage ? x.storage->data_ptr : nullptr;
        void* write_y = y.storage ? y.storage->data_ptr : nullptr;

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_A) reads.push_back(read_A);
        if (read_x) reads.push_back(read_x);
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_y) writes.push_back(write_y);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl2.gemv";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl2.gemv"_op;
        meta.set_param(0, transA);
        meta.set_param(1, alpha);
        meta.set_param(2, beta);

        auto layout = A.layout;

        return Internal::execute_blas_task(engine_, meta, reads, writes, A.dtype,
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                // oneMKL has no GEMV instantiation for sycl::half; fail loudly rather than
                // silently no-op (P1 corollary) or let it surface as a template error in MKL.
                if constexpr (std::is_same_v<T, sycl::half>)
                {
                    SB_THROW_IF(true, "oneMKL does not support HALF precision for GEMV.");
                    return sycl::event();
                }
                else
                {
                    auto pA = static_cast<const T*>(read_A);
                    auto px = static_cast<const T*>(read_x);
                    auto py = static_cast<T*>(write_y);
                    auto alpha_t = static_cast<T>(alpha);
                    auto beta_t = static_cast<T>(beta);

                    return gemv_dispatch<T>(q, layout, mkl_transA, m, n, alpha_t, pA, lda, px, incx, beta_t, py, incy, deps);
                }
            });
    }
} // namespace SushiBLAS

