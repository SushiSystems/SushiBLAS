/**************************************************************************/
/* trsv.cpp                                                               */
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
 * @file trsv.cpp
 * @brief Defines Level2::trsv, which records the BLAS TRSV task.
 * @author Mustafa Garip
 */

#include <complex>
#include <kernels/blas.hpp>
#include <SushiBLAS/engine/blas/utils.hpp>
#include <SushiBLAS/engine/blas/level2.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle Level2::trsv(const Tensor& A, Tensor& b,
                            bool upper, bool transA, bool unit_diag)
    {
        SB_THROW_IF(A.rank < 2, "TRSV requires A to be at least a 2D matrix.");
        Internal::require_level2_matrix_rank(A, "blas.lvl2.trsv");
        SB_THROW_IF(A.dtype != b.dtype, "Data type mismatch in TRSV.");

        int32_t rA = A.rank;
        Dispatch::require_storage(A, "blas.lvl2.trsv", "A");
        Dispatch::require_contiguous(A, "blas.lvl2.trsv", "A");
        int64_t n = A.shape[rA - 1]; // Triangular matrix must be n x n
        SB_THROW_IF(A.shape[rA - 2] != n, "TRSV requires A to be a square matrix.");

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];
        // b is an n x 1 matrix: ldb is 1 in row-major and n in column-major; see src/README.md.
        int64_t ldb = (A.layout == Core::Layout::ROW_MAJOR) ? 1 : n;

        int64_t nb, incb;
        Internal::get_vec_params(b, nb, incb, "blas.lvl2.trsv", "b");
        SB_THROW_IF(nb != n, "Dimension mismatch for vector b in TRSV.");
        SB_THROW_IF(incb != 1, "TRSV requires a contiguous vector b.");

        const void* read_A = A.data_or_null();
        void* rw_b = b.data_or_null(); // TRSV overwrites b

        // b arrives as the right-hand side and leaves as the solution: read, then written.
        OperandSet operands;
        operands.reads(A).updates(b);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl2.trsv";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl2.trsv"_op;
        meta.set_param(0, upper);
        meta.set_param(1, transA);
        meta.set_param(2, unit_diag);

        auto layout = A.layout;

        return Internal::execute_blas_task(recorder_, meta, operands, A.dtype,
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pA = static_cast<const T*>(read_A);
                auto pb = static_cast<T*>(rw_b);
                T one(1);

                return dispatcher.run(q, deps,
                    [=] { Kernels::Reference::trsm<T>(layout, true, upper, transA, unit_diag, n, 1, one, pA, lda, pb, ldb); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::trsm<T>(q2, layout, true, upper, transA, unit_diag, n, 1, one, pA, lda, 0, pb, ldb, 0, 1, deps2);
                    });
            });
    }
}
