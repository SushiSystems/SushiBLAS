/**************************************************************************/
/* iamax.cpp                                                              */
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
 * @file iamax.cpp
 * @brief Defines Level1::iamax, which records the BLAS IAMAX task.
 * @author Mustafa Garip
 */

#include <complex>
#include <kernels/blas.hpp>
#include <SushiBLAS/engine/blas/utils.hpp>
#include <SushiBLAS/engine/blas/level1.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // The result tensor must be INT64, because the result is an index; see src/README.md.
    TaskHandle Level1::iamax(const Tensor& x, Tensor& result)
    {
        Internal::require_scalar_result(result, Core::DataType::INT64, "blas.lvl1.iamax");

        int64_t n, incx;
        Internal::get_vec_params(x, n, incx, "blas.lvl1.iamax", "x");

        const void* read_x = x.data_or_null();
        void* write_r = result.data_or_null();

        OperandSet operands;
        operands.reads(x).writes(result);

        const ScratchBuffer scratch = Internal::allocate_level1_scratch(
            recorder_, x, static_cast<std::size_t>(n),
            Kernels::SYCL::K_LEVEL1_PARTIAL_BYTES, operands);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl1.iamax";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl1.iamax"_op;

        return Internal::execute_blas_task(recorder_, meta, operands, x.dtype,
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pX = static_cast<const T*>(read_x);
                auto pR = static_cast<int64_t*>(write_r);

                return dispatcher.run(q, deps,
                    [=] { Kernels::Reference::iamax<T>(n, pX, incx, pR); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::iamax<T>(q2, n, pX, incx, pR, scratch.data(), scratch.bytes(), deps2);
                    });
            });
    }
}
