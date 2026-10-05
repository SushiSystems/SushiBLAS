/****************************************************************/
/* trmv.cpp                                                     */
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

    sycl::event Level2::trmv(const Tensor& A, Tensor& x,
                             bool upper, bool transA, bool unit_diag)
    {
        SB_THROW_IF(A.rank < 2, "TRMV requires A to be at least a 2D matrix.");
        SB_THROW_IF(A.dtype != x.dtype, "Data type mismatch in TRMV.");

        int32_t rA = A.rank;
        int64_t n = A.shape[rA - 1]; // Triangular matrix must be n x n
        SB_THROW_IF(A.shape[rA - 2] != n, "TRMV requires A to be a square matrix.");

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];

        int64_t nx, incx;
        Internal::get_vec_params(x, nx, incx);

        SB_THROW_IF(nx != n, "Dimension mismatch for vector x in TRMV.");

        // TODO: Implement multi-dimensional batch support for Level 2

        void* read_A = A.data_or_null();
        void* rw_x = x.data_or_null(); // TRMV overwrites x

        // `x = op(A) * x` is computed in place, so x is read as well as written.
        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_A) reads.push_back(A.region());
        if (rw_x) reads.push_back(x.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (rw_x) writes.push_back(x.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl2.trmv";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl2.trmv"_op;
        meta.set_param(0, upper);
        meta.set_param(1, transA);
        meta.set_param(2, unit_diag);

        auto layout = A.layout;
        auto mode = engine_.get_mode();

        return Internal::execute_blas_task(engine_, meta, reads, writes, A.dtype,
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pA = static_cast<const T*>(read_A);
                auto px = static_cast<T*>(rw_x);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::trmv<T>(layout, upper, transA, unit_diag, n, pA, lda, px, incx); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::trmv<T>(q2, layout, upper, transA, unit_diag, n, pA, lda, px, incx, deps2);
                    });
            });
    }
}
