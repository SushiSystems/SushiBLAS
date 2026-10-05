/****************************************************************/
/* ger.cpp                                                      */
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

    sycl::event Level2::ger(const Tensor& x, const Tensor& y, Tensor& A, double alpha)
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

        // TODO: Implement multi-dimensional batch support for Level 2

        void* read_x = x.data_or_null();
        void* read_y = y.data_or_null();
        void* rw_A = A.data_or_null();

        // A is an accumulator: the rank-1 update reads every element it writes. Ordering
        // happened to survive the old write-only declaration because the tracker inserts a
        // WAW edge anyway, but any pass that trusts the declared access set - aliasing
        // analysis, arena reuse, parallel scheduling - would reason from a false premise.
        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_x) reads.push_back(x.region());
        if (read_y) reads.push_back(y.region());
        if (rw_A) reads.push_back(A.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (rw_A) writes.push_back(A.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl2.ger";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl2.ger"_op;
        meta.set_param(0, alpha);

        auto layout = A.layout;
        auto mode = engine_.get_mode();

        return Internal::execute_blas_task(engine_, meta, reads, writes, A.dtype,
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto px = static_cast<const T*>(read_x);
                auto py = static_cast<const T*>(read_y);
                auto pA = static_cast<T*>(rw_A);
                auto alpha_t = static_cast<T>(alpha);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::ger<T>(layout, m, n, alpha_t, px, incx, py, incy, pA, lda); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::ger<T>(q2, layout, m, n, alpha_t, px, incx, py, incy, pA, lda, deps2);
                    });
            });
    }
}
