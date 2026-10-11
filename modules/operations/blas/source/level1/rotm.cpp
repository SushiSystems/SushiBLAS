/**************************************************************************/
/* rotm.cpp                                                               */
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
 * @file rotm.cpp
 * @brief Defines Level1::rotm, which records the BLAS ROTM task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blas/level1.hpp>

#include <vector>
#include <SushiBLAS/blas_kernels/blas.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/blas/blas_task.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/operands/task_metadata.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiBLAS/blas_seam_kernels/modified_givens.hpp>
#include <SushiRuntime/task/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle Level1::rotm(Tensor& x, Tensor& y, const Tensor& param)
    {
        const char* const op = "blas.lvl1.rotm";

        Dispatch::require_real_dtype(x, op, "x", "rot");
        Dispatch::require_dtype(y, x.dtype, op, "y");
        Dispatch::require_dtype(param, x.dtype, op, "param");

        const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");
        const Dispatch::VectorOperand vy = Dispatch::describe_vector_operand(y, op, "y");
        Dispatch::require_same_element_count(x, y, op, "x", "y");
        const Dispatch::VectorOperand vp = Dispatch::describe_vector_operand(param, op, "param");
        Dispatch::require_vector_length(vp, Kernels::K_MODIFIED_GIVENS_PARAMETERS,
                                        "the flag and the entries of H", op, "param");
        Dispatch::require_unit_increment(vp, op, "param");

        OperandSet operands;
        operands.updates(x).updates(y).reads(param);

        const Internal::BlasMetadata meta(op, "blas.lvl1.rotm"_op);

        const int64_t n = vx.length;
        const int64_t incx = vx.increment;
        const int64_t incy = vy.increment;
        void* rw_x = x.data_or_null();
        void* rw_y = y.data_or_null();
        const void* read_param = param.data_or_null();

        return Internal::execute_blas_task<Dispatch::Domain::Real>(
            recorder_, meta.get(), operands, x.dtype,
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto px = static_cast<T*>(rw_x);
                auto py = static_cast<T*>(rw_y);
                auto pp = static_cast<const T*>(read_param);

                return dispatcher.run(q, deps,
                    [=] { Kernels::Reference::rotm<T>(n, px, incx, py, incy, pp); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::rotm<T>(q2, n, px, incx, py, incy, pp, deps2);
                    });
            });
    }
} // namespace SushiBLAS
