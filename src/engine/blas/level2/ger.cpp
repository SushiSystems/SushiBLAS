/**************************************************************************/
/* ger.cpp                                                                */
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
 * @file ger.cpp
 * @brief Defines Level2::ger, which records the BLAS GER task.
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

    TaskHandle Level2::ger(const Tensor& x, const Tensor& y, Tensor& A, double alpha)
    {
        SB_THROW_IF(A.rank < 2, "GER requires A to be at least a 2D matrix.");
        Internal::require_level2_matrix_rank(A, "blas.lvl2.ger");
        SB_THROW_IF(A.dtype != x.dtype || A.dtype != y.dtype, "Data type mismatch in GER.");

        int32_t rA = A.rank;
        Dispatch::require_storage(A, "blas.lvl2.ger", "A");
        Dispatch::require_contiguous(A, "blas.lvl2.ger", "A");
        int64_t m = A.shape[rA - 2];
        int64_t n = A.shape[rA - 1];

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];

        int64_t nx, incx;
        Internal::get_vec_params(x, nx, incx, "blas.lvl2.ger", "x");

        int64_t ny, incy;
        Internal::get_vec_params(y, ny, incy, "blas.lvl2.ger", "y");

        SB_THROW_IF(nx != m, "Dimension mismatch for vector x in GER.");
        SB_THROW_IF(ny != n, "Dimension mismatch for vector y in GER.");

        const void* read_x = x.data_or_null();
        const void* read_y = y.data_or_null();
        void* rw_A = A.data_or_null();

        // A is read as well as written by the rank-1 update; see src/README.md.
        OperandSet operands;
        operands.reads(x).reads(y).updates(A);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl2.ger";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl2.ger"_op;
        meta.set_param(0, alpha);

        auto layout = A.layout;

        return Internal::execute_blas_task(recorder_, meta, operands, A.dtype,
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto px = static_cast<const T*>(read_x);
                auto py = static_cast<const T*>(read_y);
                auto pA = static_cast<T*>(rw_A);
                auto alpha_t = static_cast<T>(alpha);

                return dispatcher.run(q, deps,
                    [=] { Kernels::Reference::ger<T>(layout, m, n, alpha_t, px, incx, py, incy, pA, lda); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::ger<T>(q2, layout, m, n, alpha_t, px, incx, py, incy, pA, lda, deps2);
                    });
            });
    }
}
