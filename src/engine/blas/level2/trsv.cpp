/****************************************************************/
/* trsv.cpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// TRSV is a single-RHS-column TRSM (n=1), so it dispatches straight into the
// TRSM kernels rather than duplicating the substitution algorithm - see
// kernels/{reference,sycl}/blas/level3/trsm.hpp.

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

    sycl::event Level2::trsv(const Tensor& A, Tensor& b,
                             bool upper, bool transA, bool unit_diag)
    {
        SB_THROW_IF(A.rank < 2, "TRSV requires A to be at least a 2D matrix.");
        SB_THROW_IF(A.dtype != b.dtype, "Data type mismatch in TRSV.");

        int32_t rA = A.rank;
        int64_t n = A.shape[rA - 1]; // Triangular matrix must be n x n
        SB_THROW_IF(A.shape[rA - 2] != n, "TRSV requires A to be a square matrix.");

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];
        // b is treated as an n x 1 "matrix": a single row-major row has 1
        // element, so ldb=1; a single column-major column just needs ldb>=n.
        int64_t ldb = (A.layout == Core::Layout::ROW_MAJOR) ? 1 : n;

        int64_t nb, incb;
        Internal::get_vec_params(b, nb, incb);
        SB_THROW_IF(nb != n, "Dimension mismatch for vector b in TRSV.");
        SB_THROW_IF(incb != 1, "TRSV requires a contiguous vector b.");

        // TODO: Implement multi-dimensional batch support for Level 2

        void* read_A = A.data_or_null();
        void* rw_b = b.data_or_null(); // TRSV overwrites b

        // b arrives as the right-hand side and leaves as the solution: read, then written.
        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_A) reads.push_back(A.region());
        if (rw_b) reads.push_back(b.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (rw_b) writes.push_back(b.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl2.trsv";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl2.trsv"_op;
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
                auto pb = static_cast<T*>(rw_b);
                T one(1);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::trsm<T>(layout, true, upper, transA, unit_diag, n, 1, one, pA, lda, pb, ldb); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::trsm<T>(q2, layout, true, upper, transA, unit_diag, n, 1, one, pA, lda, 0, pb, ldb, 0, 1, deps2);
                    });
            });
    }
}
