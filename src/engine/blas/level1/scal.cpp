/**************************************************************************/
/* scal.cpp                                                               */
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
 * @file scal.cpp
 * @brief Defines Level1::scal, which records the BLAS SCAL task.
 * @author Mustafa Garip
 */

#include <complex>
#include <kernels/blas.hpp>
#include <SushiBLAS/engine/blas/utils.hpp>
#include <SushiBLAS/engine/blas/level1.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle Level1::scal(double alpha, Tensor& x)
    {
        int64_t n, incx;
        Internal::get_vec_params(x, n, incx, "blas.lvl1.scal", "x");

        void* rw_x = x.data_or_null();

        // In place: `x = alpha * x` reads every element it writes.
        OperandSet operands;
        operands.updates(x);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl1.scal";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl1.scal"_op;
        meta.set_param(0, alpha);

        return Internal::execute_blas_task(recorder_, meta, operands, x.dtype,
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pX = static_cast<T*>(rw_x);
                auto alpha_t = static_cast<T>(alpha);

                return dispatcher.run(q, deps,
                    [=] { Kernels::Reference::scal<T>(n, alpha_t, pX, incx); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::scal<T>(q2, n, alpha_t, pX, incx, deps2);
                    });
            });
    }
}
