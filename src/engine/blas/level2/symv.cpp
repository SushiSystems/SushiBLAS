/**************************************************************************/
/* symv.cpp                                                               */
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
 * @file symv.cpp
 * @brief Defines Level2::symv, which records the BLAS SYMV task.
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

    TaskHandle Level2::symv(const Tensor& A, const Tensor& x, Tensor& y,
                            bool upper, double alpha, double beta)
    {
        SB_THROW_IF(A.rank < 2, "SYMV requires A to be at least a 2D matrix.");
        Internal::require_level2_matrix_rank(A, "blas.lvl2.symv");
        SB_THROW_IF(A.dtype != x.dtype || A.dtype != y.dtype, "Data type mismatch in SYMV.");

        int32_t rA = A.rank;
        Dispatch::require_storage(A, "blas.lvl2.symv", "A");
        Dispatch::require_contiguous(A, "blas.lvl2.symv", "A");
        int64_t n = A.shape[rA - 1]; // Symmetric matrix must be n x n
        SB_THROW_IF(A.shape[rA - 2] != n, "SYMV requires A to be a square matrix.");

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];

        int64_t nx, incx;
        Internal::get_vec_params(x, nx, incx, "blas.lvl2.symv", "x");

        int64_t ny, incy;
        Internal::get_vec_params(y, ny, incy, "blas.lvl2.symv", "y");

        SB_THROW_IF(nx != n, "Dimension mismatch for vector x in SYMV.");
        SB_THROW_IF(ny != n, "Dimension mismatch for vector y in SYMV.");

        const void* read_A = A.data_or_null();
        const void* read_x = x.data_or_null();
        void* rw_y = y.data_or_null();

        // Same beta rule as GEMV: y is read back only when beta is non-zero.
        OperandSet operands;
        operands.reads(A).reads(x);
        if (beta != 0.0)
            operands.reads(y);
        operands.writes(y);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl2.symv";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl2.symv"_op;
        meta.set_param(0, upper);
        meta.set_param(1, alpha);
        meta.set_param(2, beta);

        auto layout = A.layout;

        return Internal::execute_blas_task(recorder_, meta, operands, A.dtype,
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pA = static_cast<const T*>(read_A);
                auto px = static_cast<const T*>(read_x);
                auto py = static_cast<T*>(rw_y);
                auto alpha_t = static_cast<T>(alpha);
                auto beta_t = static_cast<T>(beta);

                return dispatcher.run(q, deps,
                    [=] { Kernels::Reference::symv<T>(layout, upper, n, alpha_t, pA, lda, px, incx, beta_t, py, incy); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::symv<T>(q2, layout, upper, n, alpha_t, pA, lda, px, incx, beta_t, py, incy, deps2);
                    });
            });
    }
}
