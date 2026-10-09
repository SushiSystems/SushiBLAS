/**************************************************************************/
/* trsm.cpp                                                               */
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
 * @file trsm.cpp
 * @brief Defines Level3::trsm, which records the BLAS TRSM task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blas/level3.hpp>

#include <vector>
#include <complex>
#include <cstddef>
#include <SushiBLAS/blas_kernels/blas.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/operands/utils.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/operands/task_metadata.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/matrix_operand.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle Level3::trsm(const Tensor& A, Tensor& B, Core::Side side, Core::Uplo uplo,
                            Core::Transpose trans, Core::Diag diag, BlasScalar alpha)
    {
        const char* const op = "blas.lvl3.trsm";

        Dispatch::require_flag(side, op, "side");
        Dispatch::require_flag(uplo, op, "uplo");
        Dispatch::require_flag(trans, op, "trans");
        Dispatch::require_flag(diag, op, "diag");
        Dispatch::require_dtype(B, A.dtype, op, "B");
        Dispatch::require_real_scalar(alpha, A.dtype, op, "alpha");

        const Dispatch::MatrixOperand a = Dispatch::describe_matrix_operand(A, op, "A");
        const Dispatch::MatrixOperand b = Dispatch::describe_matrix_operand(B, op, "B");
        Dispatch::require_same_layout(a, b, op, "A", "B");
        Dispatch::require_square(a, op, "A");

        const bool left = (side == Core::Side::Left);
        const int64_t m = b.rows;
        const int64_t n = b.cols;
        Dispatch::require_matrix_extent(a.rows, left ? m : n,
                                        left ? "rows against the rows of B"
                                             : "rows against the columns of B",
                                        op, "A");

        const int64_t batch_size = b.batch_count;
        const int64_t str_a = Dispatch::batch_stride(a, batch_size, op, "A");
        const int64_t str_b = Dispatch::batch_stride(b, batch_size, op, "B");
        Dispatch::require_same_batch_axes(A, a, B, b, op, "A", "B");

        // B arrives as the right-hand sides and leaves as the solution X: read, then written.
        OperandSet operands;
        operands.reads(A).updates(B);
        const ScratchBuffer work = Internal::allocate_blas_scratch(
            recorder_, B,
            Internal::running_solution_bytes(A.dtype) *
                static_cast<std::size_t>(batch_size * m * n),
            operands);

        const Internal::BlasMetadata meta = Internal::BlasMetadata(op, "blas.lvl3.trsm"_op)
                                                .scalar(alpha)
                                                .flag(side).flag(uplo).flag(trans).flag(diag);

        const Core::Layout layout = b.layout;
        const int64_t lda = a.leading_dimension;
        const int64_t ldb = b.leading_dimension;
        const void* read_A = A.data_or_null();
        void* rw_B = B.data_or_null();

        return Internal::execute_blas_task(recorder_, meta.get(), operands, A.dtype,
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pA = static_cast<const T*>(read_A);
                auto pB = static_cast<T*>(rw_B);
                const T alpha_t = alpha.as<T>();

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::trsm<T>(layout, side, uplo, trans, diag, m, n, alpha_t,
                                                    pA, lda, str_a, pB, ldb, str_b, batch_size);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::trsm<T>(
                            q2, layout, side, uplo, trans, diag, m, n, alpha_t, pA, lda, str_a,
                            pB, ldb, str_b, batch_size,
                            static_cast<Kernels::Accumulator<T>*>(work.data()), deps2);
                    });
            });
    }
} // namespace SushiBLAS
