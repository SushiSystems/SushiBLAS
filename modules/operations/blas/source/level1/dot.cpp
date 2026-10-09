/**************************************************************************/
/* dot.cpp                                                                */
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
 * @file dot.cpp
 * @brief Defines Level1::dot, dotu and dotc, which record the BLAS DOT, DOTU and DOTC tasks.
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
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /**
         * @brief Records result = x^T * y, or x^H * y under the conjugating policy.
         * @tparam Conj Kernels::Unconjugated for DOT and DOTU, Kernels::Conjugated for DOTC.
         */
        template<typename Conj>
        TaskHandle record_dot(TaskRecorder& recorder, const char* op,
                              SushiRuntime::Graph::OpID op_id,
                              const Tensor& x, const Tensor& y, Tensor& result)
        {
            Dispatch::require_dtype(y, x.dtype, op, "y");
            const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");
            const Dispatch::VectorOperand vy = Dispatch::describe_vector_operand(y, op, "y");
            Dispatch::require_same_element_count(x, y, op, "x", "y");
            Internal::require_scalar_result(result, x.dtype, op);

            const int64_t n = vx.length;

            OperandSet operands;
            operands.reads(x).reads(y).writes(result);

            const ScratchBuffer scratch = Internal::allocate_level1_scratch(
                recorder, x, static_cast<std::size_t>(n),
                Kernels::SYCL::K_LEVEL1_PARTIAL_BYTES, operands);

            const Internal::BlasMetadata meta(op, op_id);

            const int64_t incx = vx.increment;
            const int64_t incy = vy.increment;
            const void* read_x = x.data_or_null();
            const void* read_y = y.data_or_null();
            void* write_r = result.data_or_null();

            return Internal::execute_blas_task(recorder, meta.get(), operands, x.dtype,
                [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                    sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    using T = decltype(scalar_type);
                    using Policy = Kernels::ConjugationFor<T, Conj>;
                    auto pX = static_cast<const T*>(read_x);
                    auto pY = static_cast<const T*>(read_y);
                    auto pR = static_cast<T*>(write_r);

                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::dot<T, Policy>(n, pX, incx, pY, incy, pR);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::dot<T, Policy>(q2, n, pX, incx, pY, incy, pR,
                                                                 scratch.data(), scratch.bytes(),
                                                                 deps2);
                        });
                });
        }
    } // namespace

    TaskHandle Level1::dot(const Tensor& x, const Tensor& y, Tensor& result)
    {
        const char* const op = "blas.lvl1.dot";

        Dispatch::require_real_dtype(x, op, "x", "dotu or dotc");
        return record_dot<Kernels::Unconjugated>(recorder_, op, "blas.lvl1.dot"_op, x, y, result);
    }

    TaskHandle Level1::dotu(const Tensor& x, const Tensor& y, Tensor& result)
    {
        if (!Core::is_complex_dtype(x.dtype))
            return dot(x, y, result);

        return record_dot<Kernels::Unconjugated>(recorder_, "blas.lvl1.dotu", "blas.lvl1.dotu"_op,
                                                 x, y, result);
    }

    TaskHandle Level1::dotc(const Tensor& x, const Tensor& y, Tensor& result)
    {
        if (!Core::is_complex_dtype(x.dtype))
            return dot(x, y, result);

        return record_dot<Kernels::Conjugated>(recorder_, "blas.lvl1.dotc", "blas.lvl1.dotc"_op,
                                               x, y, result);
    }
} // namespace SushiBLAS
