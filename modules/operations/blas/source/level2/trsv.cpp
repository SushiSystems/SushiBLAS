/**************************************************************************/
/* trsv.cpp                                                               */
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
 * @file trsv.cpp
 * @brief Defines Level2::trsv, which records the BLAS TRSV task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blas/level2.hpp>

#include <vector>
#include <cstddef>
#include <complex>
#include <SushiBLAS/blas_kernels/blas.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/blas/blas_task.hpp>
#include <SushiBLAS/operands/scratch_lease.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/operands/task_metadata.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/matrix_operand.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiRuntime/task/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle Level2::trsv(const Tensor& A, Tensor& b, Core::Uplo uplo, Core::Transpose trans,
                            Core::Diag diag)
    {
        const char* const op = "blas.lvl2.trsv";

        Dispatch::require_flag(uplo, op, "uplo");
        Dispatch::require_flag(trans, op, "trans");
        Dispatch::require_flag(diag, op, "diag");
        Dispatch::require_dtype(b, A.dtype, op, "b");

        const Dispatch::MatrixOperand a = Dispatch::describe_matrix_operand(A, op, "A");
        Dispatch::require_unbatched(a, op, "A");
        Dispatch::require_square(a, op, "A");
        const Dispatch::VectorOperand vb = Dispatch::describe_vector_operand(b, op, "b");
        Dispatch::require_unit_increment(vb, op, "b");
        Dispatch::require_vector_length(vb, a.cols, "the order of A", op, "b");

        const int64_t n = a.cols;

        OperandSet operands;
        operands.reads(A).updates(b);
        const ScratchBuffer work = Internal::allocate_blas_scratch(
            recorder_, b,
            Internal::running_solution_bytes(A.dtype) * static_cast<std::size_t>(n), operands);

        const Internal::BLASMetadata meta =
            Internal::BLASMetadata(op, "blas.lvl2.trsv"_op).flag(uplo).flag(trans).flag(diag);

        const Core::Layout layout = a.layout;
        const int64_t lda = a.leading_dimension;
        // b is an n x 1 matrix: ldb is 1 in row-major and n in column-major; see modules/operations/README.md.
        const int64_t ldb = (layout == Core::Layout::RowMajor) ? 1 : n;
        const void* read_A = A.data_or_null();
        void* rw_b = b.data_or_null();

        return Internal::execute_blas_task(recorder_, meta.get(), operands, A.dtype,
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pA = static_cast<const T*>(read_A);
                auto pb = static_cast<T*>(rw_b);

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::trsm<T>(layout, Core::Side::Left, uplo, trans, diag,
                                                    n, 1, T(1), pA, lda, pb, ldb);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::trsm<T>(
                            q2, layout, Core::Side::Left, uplo, trans, diag, n, 1, T(1), pA, lda,
                            0, pb, ldb, 0, 1,
                            static_cast<Kernels::Accumulator<T>*>(work.data()), deps2);
                    });
            });
    }
} // namespace SushiBLAS
