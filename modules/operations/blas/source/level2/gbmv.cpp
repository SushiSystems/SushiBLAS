/**************************************************************************/
/* gbmv.cpp                                                               */
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
 * @file gbmv.cpp
 * @brief Defines Level2::gbmv, which records the BLAS GBMV task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blas/level2.hpp>

#include <vector>
#include <complex>
#include <SushiBLAS/blas_kernels/blas.hpp>
#include <SushiBLAS/blas/blas_task.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/blas/band_operand.hpp>
#include <SushiBLAS/operands/task_metadata.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiRuntime/task/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle Level2::gbmv(const Tensor& A, const Tensor& x, Tensor& y, int64_t kl, int64_t ku,
                            Core::Transpose trans, BlasScalar alpha, BlasScalar beta)
    {
        const char* const op = "blas.lvl2.gbmv";

        Dispatch::require_flag(trans, op, "trans");
        Dispatch::require_dtype(x, A.dtype, op, "x");
        Dispatch::require_dtype(y, A.dtype, op, "y");
        Dispatch::require_real_scalar(alpha, A.dtype, op, "alpha");
        Dispatch::require_real_scalar(beta, A.dtype, op, "beta");

        const Dispatch::BandOperand a = Dispatch::describe_band_operand(A, kl, ku, op, "A");
        const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");
        const Dispatch::VectorOperand vy = Dispatch::describe_vector_operand(y, op, "y");

        // The band storage fixes one extent of A; the vector along the other axis gives the rest.
        const bool transposed = Core::is_transposed(trans);
        if (a.layout == Core::Layout::ColumnMajor)
            Dispatch::require_vector_length(transposed ? vy : vx, a.lines, "the columns of A", op,
                                            transposed ? "y" : "x");
        else
            Dispatch::require_vector_length(transposed ? vx : vy, a.lines, "the rows of A", op,
                                            transposed ? "x" : "y");
        const int64_t m = transposed ? vx.length : vy.length;
        const int64_t n = transposed ? vy.length : vx.length;

        OperandSet operands;
        operands.reads(A).reads(x);
        if (beta.is_zero())
            operands.writes(y);
        else
            operands.updates(y);

        const Internal::BlasMetadata meta = Internal::BlasMetadata(op, "blas.lvl2.gbmv"_op)
                                                .scalar(alpha).scalar(beta)
                                                .flag(trans).extent(kl).extent(ku);

        const Core::Layout layout = a.layout;
        const int64_t lda = a.leading_dimension;
        const int64_t incx = vx.increment;
        const int64_t incy = vy.increment;
        const void* read_A = A.data_or_null();
        const void* read_x = x.data_or_null();
        void* rw_y = y.data_or_null();

        return Internal::execute_blas_task(recorder_, meta.get(), operands, A.dtype,
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pA = static_cast<const T*>(read_A);
                auto px = static_cast<const T*>(read_x);
                auto py = static_cast<T*>(rw_y);
                const T alpha_t = alpha.as<T>();
                const T beta_t = beta.as<T>();

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::gbmv<T>(layout, trans, m, n, kl, ku, alpha_t, pA, lda,
                                                    px, incx, beta_t, py, incy);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::gbmv<T>(q2, layout, trans, m, n, kl, ku, alpha_t,
                                                      pA, lda, px, incx, beta_t, py, incy, deps2);
                    });
            });
    }
} // namespace SushiBLAS
