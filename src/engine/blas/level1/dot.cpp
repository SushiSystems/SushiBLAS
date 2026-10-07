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
 * @brief Defines Level1::dot, which records the BLAS DOT task.
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

    TaskHandle Level1::dot(const Tensor& x, const Tensor& y, Tensor& result)
    {
        SB_THROW_IF(x.dtype != y.dtype, "Data type mismatch in DOT.");
        // A complex dtype computes the conjugated dot product (DOTC); see src/README.md.

        int64_t n, incx, incy;
        Internal::get_vec_params(x, n, incx, "blas.lvl1.dot", "x");
        int64_t ny;
        Internal::get_vec_params(y, ny, incy, "blas.lvl1.dot", "y");
        SB_THROW_IF(n != ny, "DOT requires tensors of the same number of elements.");
        Internal::require_scalar_result(result, x.dtype, "blas.lvl1.dot");

        const void* read_x = x.data_or_null();
        const void* read_y = y.data_or_null();
        void* write_r = result.data_or_null();

        OperandSet operands;
        operands.reads(x).reads(y).writes(result);

        const ScratchBuffer scratch = Internal::allocate_level1_scratch(
            recorder_, x, static_cast<std::size_t>(n),
            Kernels::SYCL::K_LEVEL1_PARTIAL_BYTES, operands);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl1.dot";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl1.dot"_op;

        return Internal::execute_blas_task(recorder_, meta, operands, x.dtype,
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pX = static_cast<const T*>(read_x);
                auto pY = static_cast<const T*>(read_y);
                auto pR = static_cast<T*>(write_r);

                if constexpr (Core::is_complex_v<T>)
                {
                    return dispatcher.run(q, deps,
                        [=] { Kernels::Reference::dotc<typename T::value_type>(n, pX, incx, pY, incy, pR); },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::dotc<typename T::value_type>(q2, n, pX, incx, pY, incy, pR, scratch.data(), scratch.bytes(), deps2);
                        });
                }
                else
                {
                    return dispatcher.run(q, deps,
                        [=] { Kernels::Reference::dot<T>(n, pX, incx, pY, incy, pR); },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::dot<T>(q2, n, pX, incx, pY, incy, pR, scratch.data(), scratch.bytes(), deps2);
                        });
                }
            });
    }
} // namespace SushiBLAS
