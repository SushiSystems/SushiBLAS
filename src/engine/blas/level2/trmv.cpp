/**************************************************************************/
/* trmv.cpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file trmv.cpp
 * @brief Defines Level2::trmv, which records the BLAS TRMV task.
 * @author Mustafa Garip
 */

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
        Internal::require_level2_matrix_rank(A, "blas.lvl2.trmv");
        SB_THROW_IF(A.dtype != x.dtype, "Data type mismatch in TRMV.");

        int32_t rA = A.rank;
        Dispatch::require_storage(A, "blas.lvl2.trmv", "A");
        Dispatch::require_contiguous(A, "blas.lvl2.trmv", "A");
        int64_t n = A.shape[rA - 1]; // Triangular matrix must be n x n
        SB_THROW_IF(A.shape[rA - 2] != n, "TRMV requires A to be a square matrix.");

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];

        int64_t nx, incx;
        Internal::get_vec_params(x, nx, incx, "blas.lvl2.trmv", "x");

        SB_THROW_IF(nx != n, "Dimension mismatch for vector x in TRMV.");

        const void* read_A = A.data_or_null();
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

        return Internal::execute_blas_task(engine_, meta, reads, writes, A.dtype,
            Internal::ReferenceOperands{read_A, rw_x},
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pA = static_cast<const T*>(read_A);
                auto px = static_cast<T*>(rw_x);

                return dispatcher.run(q, deps,
                    [=] { Kernels::Reference::trmv<T>(layout, upper, transA, unit_diag, n, pA, lda, px, incx); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::trmv<T>(q2, layout, upper, transA, unit_diag, n, pA, lda, px, incx, deps2);
                    });
            });
    }
}
