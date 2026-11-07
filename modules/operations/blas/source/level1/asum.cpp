/**************************************************************************/
/* asum.cpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file asum.cpp
 * @brief Defines Level1::asum, which records the BLAS ASUM task.
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
#include <SushiBLAS/operands/blas_metadata.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    TaskHandle Level1::asum(const Tensor& x, Tensor& result)
    {
        const char* const op = "blas.lvl1.asum";

        const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");
        Internal::require_scalar_result(result, Internal::real_result_dtype(x.dtype), op);

        const int64_t n = vx.length;

        OperandSet operands;
        operands.reads(x).writes(result);

        const ScratchBuffer scratch = Internal::allocate_level1_scratch(
            recorder_, x, static_cast<std::size_t>(n),
            Kernels::SYCL::K_LEVEL1_PARTIAL_BYTES, operands);

        const Internal::BLASMetadata metadata =
            Internal::BLASMetadata(op, "blas.lvl1.asum"_op).shape(
                ReductionShape{n, 1, x.dtype});

        const int64_t incx = vx.increment;
        const void* read_x = x.data_or_null();
        void* write_r = result.data_or_null();

        return Internal::execute_blas_task(recorder_, metadata.get(), operands, x.dtype,
            [=](auto scalar_type, const KernelPairDispatcher& dispatcher,
                const std::vector<sycl::event>& deps) -> Device::SubmittedEvents
            {
                using T = decltype(scalar_type);
                using Real = Kernels::RealType<T>;
                auto pX = static_cast<const T*>(read_x);
                auto pR = static_cast<Real*>(write_r);

                return dispatcher.run(deps,
                    [=]
                    {
                        Kernels::Reference::asum<T>(n, pX, incx, pR);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> Device::SubmittedEvents
                    {
                        return Kernels::SYCL::asum<T>(q2, n, pX, incx, pR, scratch.data(),
                                                      scratch.bytes(), deps2);
                    });
            });
    }
} // namespace SushiBLAS
