/**************************************************************************/
/* swap.cpp                                                               */
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
 * @file swap.cpp
 * @brief Defines Level1::swap, which records the BLAS SWAP task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blas/level1.hpp>

#include <vector>
#include <complex>
#include <SushiBLAS/blas_kernels/blas.hpp>
#include <SushiBLAS/blas/blas_task.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/operands/blas_metadata.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    TaskHandle Level1::swap(Tensor& x, Tensor& y)
    {
        const char* const op = "blas.lvl1.swap";

        Dispatch::require_dtype(y, x.dtype, op, "y");
        const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");
        const Dispatch::VectorOperand vy = Dispatch::describe_vector_operand(y, op, "y");
        Dispatch::require_same_element_count(x, y, op, "x", "y");

        // x and y are each read and written, so both sit in both access sets; see modules/operations/README.md.
        OperandSet operands;
        operands.updates(x).updates(y);

        const Internal::BLASMetadata metadata =
            Internal::BLASMetadata(op, "blas.lvl1.swap"_op).shape(CopyShape{x.dtype});

        const int64_t n = vx.length;
        const int64_t incx = vx.increment;
        const int64_t incy = vy.increment;
        void* rw_x = x.data_or_null();
        void* rw_y = y.data_or_null();

        return Internal::execute_blas_task(recorder_, metadata.get(), operands, x.dtype,
            [=](auto scalar_type, const KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pX = static_cast<T*>(rw_x);
                auto pY = static_cast<T*>(rw_y);

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::swap<T>(n, pX, incx, pY, incy);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::swap<T>(q2, n, pX, incx, pY, incy, deps2);
                    });
            });
    }
} // namespace SushiBLAS
