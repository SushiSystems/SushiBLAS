/**************************************************************************/
/* nrm2.cpp                                                               */
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
 * @file nrm2.cpp
 * @brief Defines Level1::nrm2, which records the BLAS NRM2 task.
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
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiRuntime/task/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle Level1::nrm2(const Tensor& x, Tensor& result)
    {
        const char* const op = "blas.lvl1.nrm2";

        const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");
        Internal::require_scalar_result(result, Internal::real_result_dtype(x.dtype), op);

        const int64_t n = vx.length;

        OperandSet operands;
        operands.reads(x).writes(result);

        // A complex element folds as two real values, so the scratch is planned for 2n.
        const ScratchBuffer scratch = Internal::allocate_level1_scratch(
            recorder_, x, static_cast<std::size_t>(n) * (Core::is_complex_dtype(x.dtype) ? 2 : 1),
            Kernels::SYCL::K_LEVEL1_PARTIAL_BYTES, operands);

        const Internal::BLASMetadata meta(op, "blas.lvl1.nrm2"_op);

        const int64_t incx = vx.increment;
        const void* read_x = x.data_or_null();
        void* write_r = result.data_or_null();

        return Internal::execute_blas_task(recorder_, meta.get(), operands, x.dtype,
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                using Real = Kernels::RealType<T>;
                auto pX = static_cast<const T*>(read_x);
                auto pR = static_cast<Real*>(write_r);

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::nrm2<T>(n, pX, incx, pR);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::nrm2<T>(q2, n, pX, incx, pR, scratch.data(),
                                                      scratch.bytes(), deps2);
                    });
            });
    }
} // namespace SushiBLAS
