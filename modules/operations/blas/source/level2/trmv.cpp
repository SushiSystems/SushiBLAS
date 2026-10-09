/**************************************************************************/
/* trmv.cpp                                                               */
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
 * @file trmv.cpp
 * @brief Defines Level2::trmv, which records the BLAS TRMV task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blas/level2.hpp>

#include <vector>
#include <cstddef>
#include <complex>
#include <SushiBLAS/blas_kernels/blas.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/blas/blas_task.hpp>
#include <SushiBLAS/operands/scratch_lease.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/operands/task_metadata.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/matrix_operand.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle Level2::trmv(const Tensor& A, Tensor& x, Core::Uplo uplo, Core::Transpose trans,
                            Core::Diag diag)
    {
        const char* const op = "blas.lvl2.trmv";

        Dispatch::require_flag(uplo, op, "uplo");
        Dispatch::require_flag(trans, op, "trans");
        Dispatch::require_flag(diag, op, "diag");
        Dispatch::require_dtype(x, A.dtype, op, "x");

        const Dispatch::MatrixOperand a = Dispatch::describe_matrix_operand(A, op, "A");
        Dispatch::require_unbatched(a, op, "A");
        Dispatch::require_square(a, op, "A");
        const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");
        Dispatch::require_vector_length(vx, a.rows, "the order of A", op, "x");

        const int64_t n = a.rows;

        // The SYCL kernel reads the old x from a snapshot while it writes the new one.
        OperandSet operands;
        operands.reads(A).updates(x);
        const ScratchBuffer snapshot = Internal::allocate_blas_scratch(
            recorder_, x, Core::bytes_per_element(A.dtype) * static_cast<std::size_t>(n), operands);

        const Internal::BlasMetadata meta = Internal::BlasMetadata(op, "blas.lvl2.trmv"_op)
                                                .flag(uplo).flag(trans).flag(diag);

        const Core::Layout layout = a.layout;
        const int64_t lda = a.leading_dimension;
        const int64_t incx = vx.increment;
        const void* read_A = A.data_or_null();
        void* rw_x = x.data_or_null();

        return Internal::execute_blas_task(recorder_, meta.get(), operands, A.dtype,
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pA = static_cast<const T*>(read_A);
                auto px = static_cast<T*>(rw_x);
                auto work = static_cast<T*>(snapshot.data());

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::trmv<T>(layout, uplo, trans, diag, n, pA, lda, px,
                                                    incx);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::trmv<T>(q2, layout, uplo, trans, diag, n, pA, lda,
                                                      px, incx, work, deps2);
                    });
            });
    }
} // namespace SushiBLAS
