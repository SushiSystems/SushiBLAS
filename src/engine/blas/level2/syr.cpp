/****************************************************************/
/* syr.cpp                                                      */
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

    sycl::event Level2::syr(const Tensor& x, Tensor& A, bool upper, double alpha)
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

        // TODO: Implement multi-dimensional batch support for Level 2

        void* read_x = x.data_or_null();
        void* rw_A = A.data_or_null();

        // A is an accumulator, and only the selected triangle is touched at all - the other
        // one has to survive the task unchanged. Both facts make the region a
        // read-modify-write rather than a write.
        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_x) reads.push_back(x.region());
        if (rw_A) reads.push_back(A.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (rw_A) writes.push_back(A.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl2.syr";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl2.syr"_op;
        meta.set_param(0, upper);
        meta.set_param(1, alpha);

        auto layout = A.layout;
        auto mode = engine_.get_mode();

        return Internal::execute_blas_task(engine_, meta, reads, writes, A.dtype,
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto px = static_cast<const T*>(read_x);
                auto pA = static_cast<T*>(rw_A);
                auto alpha_t = static_cast<T>(alpha);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::syr<T>(layout, upper, n, alpha_t, px, incx, pA, lda); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::syr<T>(q2, layout, upper, n, alpha_t, px, incx, pA, lda, deps2);
                    });
            });
    }
} // namespace SushiBLAS
