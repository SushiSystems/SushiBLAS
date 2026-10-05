/****************************************************************/
/* gemv.cpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <complex>
#include <kernels/blas.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/blas/utils.hpp>
#include <SushiBLAS/engine/blas/level2.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event Level2::gemv(const Tensor& A, const Tensor& x, Tensor& y,
                             bool transA, double alpha, double beta)
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

        void* read_A = A.data_or_null();
        void* read_x = x.data_or_null();
        void* rw_y = y.data_or_null();

        // `y = alpha * op(A) * x + beta * y` reads y whenever beta is non-zero. The test is
        // on the caller's double rather than on the narrowed element type: a beta that is
        // non-zero here but flushes to zero in the element type only ever declares an edge
        // the kernel does not need, never drops one it does.
        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_A) reads.push_back(A.region());
        if (read_x) reads.push_back(x.region());
        if (rw_y && beta != 0.0) reads.push_back(y.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (rw_y) writes.push_back(y.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl2.gemv";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl2.gemv"_op;
        meta.set_param(0, transA);
        meta.set_param(1, alpha);
        meta.set_param(2, beta);

        auto layout = A.layout;
        auto mode = engine_.get_mode();

        return Internal::execute_blas_task(engine_, meta, reads, writes, A.dtype,
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pA = static_cast<const T*>(read_A);
                auto px = static_cast<const T*>(read_x);
                auto py = static_cast<T*>(rw_y);
                auto alpha_t = static_cast<T>(alpha);
                auto beta_t = static_cast<T>(beta);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::gemv<T>(layout, transA, m, n, alpha_t, pA, lda, px, incx, beta_t, py, incy); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::gemv<T>(q2, layout, transA, m, n, alpha_t, pA, lda, px, incx, beta_t, py, incy, deps2);
                    });
            });
    }
} // namespace SushiBLAS
