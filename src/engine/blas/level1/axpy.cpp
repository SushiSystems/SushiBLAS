/**************************************************************************/
/* axpy.cpp                                                               */
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
 * @file axpy.cpp
 * @brief Defines Level1::axpy, which records the BLAS AXPY task.
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

    TaskHandle Level1::axpy(double alpha, const Tensor& x, Tensor& y)
    {
        SB_THROW_IF(x.dtype != y.dtype, "Data type mismatch in AXPY.");
        int64_t n, incx, incy;
        Internal::get_vec_params(x, n, incx, "blas.lvl1.axpy", "x");
        int64_t ny; Internal::get_vec_params(y, ny, incy, "blas.lvl1.axpy", "y");
        SB_THROW_IF(n != ny, "AXPY requires tensors of the same number of elements.");

        const void* read_x = x.data_or_null();
        void* rw_y = y.data_or_null();

        // y is read before it is written, so it sits in both access sets; see src/README.md.
        OperandSet operands;
        operands.reads(x).updates(y);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl1.axpy";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl1.axpy"_op;
        meta.set_param(0, alpha);

        return Internal::execute_blas_task(recorder_, meta, operands, x.dtype,
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pX = static_cast<const T*>(read_x);
                auto pY = static_cast<T*>(rw_y);
                auto alpha_t = static_cast<T>(alpha);

                return dispatcher.run(q, deps,
                    [=] { Kernels::Reference::axpy<T>(n, alpha_t, pX, incx, pY, incy); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::axpy<T>(q2, n, alpha_t, pX, incx, pY, incy, deps2);
                    });
            });
    }
}
