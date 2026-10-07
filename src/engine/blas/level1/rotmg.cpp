/**************************************************************************/
/* rotmg.cpp                                                              */
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
 * @file rotmg.cpp
 * @brief Defines Level1::rotmg, which records the BLAS ROTMG task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/blas/level1.hpp>

#include <vector>
#include <kernels/blas.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/blas/utils.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/blas/task_metadata.hpp>
#include <SushiBLAS/engine/blas/argument_check.hpp>
#include <SushiBLAS/engine/blas/scalar_operand.hpp>
#include <SushiBLAS/engine/blas/vector_operand.hpp>
#include <SushiBLAS/kernels/blas/modified_givens.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle Level1::rotmg(Tensor& d1, Tensor& d2, Tensor& x1, const Tensor& y1, Tensor& param)
    {
        const char* const op = "blas.lvl1.rotmg";

        Dispatch::require_real_dtype(d1, op, "d1", "rotg");
        Dispatch::require_operand(d1.dtype != Core::DataType::HALF, op, "d1",
                                  "has dtype HALF; the scale window of this routine, 2^-24 to "
                                  "2^24, does not fit the type");

        Dispatch::require_scalar_operand(d1, d1.dtype, op, "d1");
        Dispatch::require_scalar_operand(d2, d1.dtype, op, "d2");
        Dispatch::require_scalar_operand(x1, d1.dtype, op, "x1");
        Dispatch::require_scalar_operand(y1, d1.dtype, op, "y1");
        Dispatch::require_dtype(param, d1.dtype, op, "param");
        const Dispatch::VectorOperand vp = Dispatch::describe_vector_operand(param, op, "param");
        Dispatch::require_vector_length(vp, Kernels::K_MODIFIED_GIVENS_PARAMETERS,
                                        "the flag and the entries of H", op, "param");
        Dispatch::require_unit_increment(vp, op, "param");

        // param is declared read as well: the elements the flag implies keep their values.
        OperandSet operands;
        operands.updates(d1).updates(d2).updates(x1).reads(y1).updates(param);

        const Internal::BlasMetadata meta(op, "blas.lvl1.rotmg"_op);

        void* rw_d1 = d1.data_or_null();
        void* rw_d2 = d2.data_or_null();
        void* rw_x1 = x1.data_or_null();
        const void* read_y1 = y1.data_or_null();
        void* rw_param = param.data_or_null();

        return Internal::execute_blas_task<Dispatch::Domain::REAL>(
            recorder_, meta.get(), operands, d1.dtype,
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pd1 = static_cast<T*>(rw_d1);
                auto pd2 = static_cast<T*>(rw_d2);
                auto px1 = static_cast<T*>(rw_x1);
                auto py1 = static_cast<const T*>(read_y1);
                auto pp = static_cast<T*>(rw_param);

                return dispatcher.run(q, deps,
                    [=] { Kernels::Reference::rotmg<T>(pd1, pd2, px1, py1, pp); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::rotmg<T>(q2, pd1, pd2, px1, py1, pp, deps2);
                    });
            });
    }
} // namespace SushiBLAS
