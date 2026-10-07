/**************************************************************************/
/* syrk.cpp                                                               */
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
 * @file syrk.cpp
 * @brief Defines Level3::syrk, which records the BLAS SYRK task.
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

    sycl::event Level3::syrk(const Tensor& A, Tensor& C, 
                      bool upper, bool transA, 
                      double alpha, double beta) 
    {
        // 1. Validation
        SB_THROW_IF(A.rank < 2 || C.rank < 2, "SYRK requires at least 2D tensors.");
        SB_THROW_IF(A.dtype != C.dtype, "Data type mismatch in SYRK operation.");

        const char* op = "blas.lvl3.syrk";
        const Dispatch::MatrixOperand a = Dispatch::describe_matrix_operand(A, op, "A");
        const Dispatch::MatrixOperand c = Dispatch::describe_matrix_operand(C, op, "C");
        Dispatch::require_same_layout(a, c, op, "A", "C");

        const int64_t n = c.rows;
        Dispatch::require_matrix_extent(c.cols, n, "columns for a square C", op, "C");

        const int64_t k = transA ? a.rows : a.cols;
        Dispatch::require_matrix_extent(transA ? a.cols : a.rows, n, "rows in op(A)", op, "A");

        const int64_t batch_size = c.batch_count;
        const int64_t str_a = Dispatch::batch_stride(a, batch_size, op, "A");
        const int64_t str_c = Dispatch::batch_stride(c, batch_size, op, "C");
        Dispatch::require_same_batch_axes(A, a, C, c, op, "A", "C");

        const int64_t lda = a.leading_dimension;
        const int64_t ldc = c.leading_dimension;

        const auto layout = c.layout;
        const void* read_A = A.data_or_null();
        void* rw_C = C.data_or_null();

        // C is read-modify-write whatever beta is; see src/README.md.
        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_A) reads.push_back(A.region());
        if (rw_C) reads.push_back(C.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (rw_C) writes.push_back(C.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl3.syrk";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl3.syrk"_op;
        meta.set_param(0, alpha);
        meta.set_param(1, beta);
        meta.set_param(2, upper);
        meta.set_param(3, transA);

        return Internal::execute_blas_task(engine_, meta, reads, writes, A.dtype,
            Internal::ReferenceOperands{read_A, rw_C},
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto pA = static_cast<const T*>(read_A);
                auto pC = static_cast<T*>(rw_C);
                auto alpha_t = static_cast<T>(alpha);
                auto beta_t = static_cast<T>(beta);

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::syrk<T>(layout, upper, transA, n, k, alpha_t, pA, lda, str_a, beta_t, pC, ldc, str_c, batch_size);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::syrk<T>(q2, layout, upper, transA, n, k, alpha_t, pA, lda, str_a, beta_t, pC, ldc, str_c, batch_size, deps2);
                    });
            });
    }
} // namespace SushiBLAS
