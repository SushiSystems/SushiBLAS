/**************************************************************************/
/* copy.cpp                                                               */
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
 * @file copy.cpp
 * @brief Defines Level1::copy, which records the BLAS COPY task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blas/level1.hpp>

#include <vector>
#include <complex>
#include <SushiBLAS/blas_kernels/blas.hpp>
#include <SushiBLAS/operands/utils.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/operands/task_metadata.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle Level1::copy(const Tensor& x, Tensor& y)
    {
        const char* const op = "blas.lvl1.copy";

        // A dtype mismatch stays an error and names the converting copy; see modules/operations/README.md.
        if (x.dtype != y.dtype)
            Dispatch::Detail::throw_operand_error(op, "y",
                                                  "has dtype {}, but 'x' has dtype {}; BLAS copy "
                                                  "moves elements without changing their type, "
                                                  "ElementwiseOps::convert copies and converts",
                                                  Core::to_string(y.dtype),
                                                  Core::to_string(x.dtype));

        const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");
        const Dispatch::VectorOperand vy = Dispatch::describe_vector_operand(y, op, "y");
        Dispatch::require_same_element_count(x, y, op, "x", "y");

        OperandSet operands;
        operands.reads(x).writes(y);

        const Internal::BlasMetadata meta(op, "blas.lvl1.copy"_op);

        const int64_t n = vx.length;
        const int64_t incx = vx.increment;
        const int64_t incy = vy.increment;
        const void* read_x = x.data_or_null();
        void* write_y = y.data_or_null();

        // copy does no arithmetic, so it serves Domain::All; see modules/operations/README.md.
        return Internal::execute_blas_task<Dispatch::Domain::All>(
            recorder_, meta.get(), operands, x.dtype,
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pX = static_cast<const T*>(read_x);
                auto pY = static_cast<T*>(write_y);

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::copy<T>(n, pX, incx, pY, incy);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::copy<T>(q2, n, pX, incx, pY, incy, deps2);
                    });
            });
    }
} // namespace SushiBLAS
