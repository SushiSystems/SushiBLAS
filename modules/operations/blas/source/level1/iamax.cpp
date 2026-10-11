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

#include <SushiBLAS/blas/level1.hpp>

#include <vector>
#include <complex>
#include <SushiBLAS/blas_kernels/blas.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/blas/blas_task.hpp>
#include <SushiBLAS/operands/scalar_result.hpp>
#include <SushiBLAS/operands/scratch_lease.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/operands/task_metadata.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiRuntime/task/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle Level1::iamax(const Tensor& x, Tensor& result)
    {
        const char* const op = "blas.lvl1.iamax";

        // The result is an index, so its tensor is INT64 whatever x holds; see modules/operations/README.md.
        Internal::require_scalar_result(result, Core::DataType::Int64, op);
        const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");

        const int64_t n = vx.length;

        OperandSet operands;
        operands.reads(x).writes(result);

        const ScratchBuffer scratch = Internal::allocate_level1_scratch(
            recorder_, x, static_cast<std::size_t>(n),
            Kernels::SYCL::K_LEVEL1_PARTIAL_BYTES, operands);

        const Internal::BLASMetadata meta(op, "blas.lvl1.iamax"_op);

        const int64_t incx = vx.increment;
        const void* read_x = x.data_or_null();
        void* write_r = result.data_or_null();

        return Internal::execute_blas_task(recorder_, meta.get(), operands, x.dtype,
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pX = static_cast<const T*>(read_x);
                auto pR = static_cast<int64_t*>(write_r);

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::iamax<T>(n, pX, incx, pR);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::iamax<T>(q2, n, pX, incx, pR, scratch.data(),
                                                       scratch.bytes(), deps2);
                    });
            });
    }
} // namespace SushiBLAS
