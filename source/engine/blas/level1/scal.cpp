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

#include <SushiBLAS/engine/blas/level1.hpp>

#include <vector>
#include <complex>
#include <kernels/blas.hpp>
#include <SushiBLAS/engine/blas/utils.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/blas/task_metadata.hpp>
#include <SushiBLAS/engine/blas/argument_check.hpp>
#include <SushiBLAS/engine/blas/vector_operand.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle Level1::scal(BlasScalar alpha, Tensor& x)
    {
        const char* const op = "blas.lvl1.scal";

        Dispatch::require_real_scalar(alpha, x.dtype, op, "alpha");
        const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");

        // x = alpha * x reads every element it writes.
        OperandSet operands;
        operands.updates(x);

        const Internal::BlasMetadata meta =
            Internal::BlasMetadata(op, "blas.lvl1.scal"_op).scalar(alpha);

        const int64_t n = vx.length;
        const int64_t incx = vx.increment;
        void* rw_x = x.data_or_null();

        return Internal::execute_blas_task(recorder_, meta.get(), operands, x.dtype,
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pX = static_cast<T*>(rw_x);
                const T alpha_t = alpha.as<T>();

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::scal<T>(n, alpha_t, pX, incx);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::scal<T>(q2, n, alpha_t, pX, incx, deps2);
                    });
            });
    }
} // namespace SushiBLAS
