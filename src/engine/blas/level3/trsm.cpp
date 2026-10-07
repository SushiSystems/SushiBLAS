/**************************************************************************/
/* trsm.cpp                                                               */
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
 * @file trsm.cpp
 * @brief Defines Level3::trsm, which records the BLAS TRSM task.
 * @author Mustafa Garip
 */

#include <vector>
#include <complex>
#include <kernels/blas.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/blas/utils.hpp>
#include <SushiBLAS/engine/blas/level3.hpp>
#include <SushiBLAS/engine/blas/matrix_operand.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event Level3::trsm(const Tensor& A, Tensor& B,
                      bool left_side, bool upper,
                      bool transA, bool unit_diag,
                      double alpha)
    {
        // 1. Validation
        SB_THROW_IF(A.rank < 2 || B.rank < 2, "TRSM requires at least 2D tensors.");
        SB_THROW_IF(A.dtype != B.dtype, "Data type mismatch in TRSM operation.");

        const char* op = "blas.lvl3.trsm";
        const Dispatch::MatrixOperand a = Dispatch::describe_matrix_operand(A, op, "A");
        const Dispatch::MatrixOperand b = Dispatch::describe_matrix_operand(B, op, "B");
        Dispatch::require_same_layout(a, b, op, "A", "B");

        const int64_t m = b.rows;
        const int64_t n = b.cols;
        Dispatch::require_matrix_extent(a.cols, a.rows, "columns for a square A", op, "A");
        Dispatch::require_matrix_extent(a.rows, left_side ? m : n,
                                        left_side ? "rows against the rows of B"
                                                  : "rows against the columns of B",
                                        op, "A");

        const int64_t batch_size = b.batch_count;
        const int64_t str_a = Dispatch::batch_stride(a, batch_size, op, "A");
        const int64_t str_b = Dispatch::batch_stride(b, batch_size, op, "B");
        Dispatch::require_same_batch_axes(A, a, B, b, op, "A", "B");

        const int64_t lda = a.leading_dimension;
        const int64_t ldb = b.leading_dimension;

        const auto layout = b.layout;
        const void* read_A = A.data_or_null();
        void* rw_B = B.data_or_null();

        // B arrives as the right-hand sides and leaves as the solution X: read, then written.
        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_A) reads.push_back(A.region());
        if (rw_B) reads.push_back(B.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (rw_B) writes.push_back(B.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl3.trsm";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl3.trsm"_op;
        meta.set_param(0, alpha);
        meta.set_param(1, left_side);
        meta.set_param(2, upper);
        meta.set_param(3, transA);
        meta.set_param(4, unit_diag);

        return Internal::execute_blas_task(engine_, meta, reads, writes, A.dtype,
            Internal::ReferenceOperands{read_A, rw_B},
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pA = static_cast<const T*>(read_A);
                auto pB = static_cast<T*>(rw_B);
                auto alpha_t = static_cast<T>(alpha);

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::trsm<T>(layout, left_side, upper, transA, unit_diag, m, n, alpha_t, pA, lda, str_a, pB, ldb, str_b, batch_size);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::trsm<T>(q2, layout, left_side, upper, transA, unit_diag, m, n, alpha_t, pA, lda, str_a, pB, ldb, str_b, batch_size, deps2);
                    });
            });
    }
} // namespace SushiBLAS
