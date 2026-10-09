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

#include <SushiBLAS/blas/level1.hpp>

#include <vector>
#include <complex>
#include <SushiBLAS/blas_kernels/blas.hpp>
#include <SushiBLAS/operands/utils.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/operands/task_metadata.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle Level1::axpy(BlasScalar alpha, const Tensor& x, Tensor& y)
    {
        const char* const op = "blas.lvl1.axpy";

        Dispatch::require_dtype(y, x.dtype, op, "y");
        Dispatch::require_real_scalar(alpha, x.dtype, op, "alpha");
        const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");
        const Dispatch::VectorOperand vy = Dispatch::describe_vector_operand(y, op, "y");
        Dispatch::require_same_element_count(x, y, op, "x", "y");

        // y is read before it is written, so it sits in both access sets; see modules/operations/README.md.
        OperandSet operands;
        operands.reads(x).updates(y);

        const Internal::BlasMetadata meta =
            Internal::BlasMetadata(op, "blas.lvl1.axpy"_op).scalar(alpha);

        const int64_t n = vx.length;
        const int64_t incx = vx.increment;
        const int64_t incy = vy.increment;
        const void* read_x = x.data_or_null();
        void* rw_y = y.data_or_null();

        return Internal::execute_blas_task(recorder_, meta.get(), operands, x.dtype,
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pX = static_cast<const T*>(read_x);
                auto pY = static_cast<T*>(rw_y);
                const T alpha_t = alpha.as<T>();

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::axpy<T>(n, alpha_t, pX, incx, pY, incy);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::axpy<T>(q2, n, alpha_t, pX, incx, pY, incy, deps2);
                    });
            });
    }
} // namespace SushiBLAS
