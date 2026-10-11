/**************************************************************************/
/* tbsv.cpp                                                               */
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
 * @file tbsv.cpp
 * @brief Defines Level2::tbsv, which records the BLAS TBSV task.
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
#include <SushiBLAS/blas/band_operand.hpp>
#include <SushiBLAS/operands/task_metadata.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    TaskHandle Level2::tbsv(const Tensor& A, Tensor& b, Core::Uplo uplo, int64_t k,
                            Core::Transpose trans, Core::Diag diag)
    {
        const char* const op = "blas.lvl2.tbsv";

        Dispatch::require_flag(uplo, op, "uplo");
        Dispatch::require_flag(trans, op, "trans");
        Dispatch::require_flag(diag, op, "diag");
        Dispatch::require_dtype(b, A.dtype, op, "b");

        const Dispatch::BandOperand a =
            Dispatch::describe_triangle_band_operand(A, uplo, k, op, "A");
        const Dispatch::VectorOperand vb = Dispatch::describe_vector_operand(b, op, "b");
        Dispatch::require_vector_length(vb, a.lines, "the order of A", op, "b");

        const int64_t n = a.lines;
        const int64_t incb = vb.increment;

        OperandSet operands;
        operands.reads(A).updates(b);

        // The wide solution is addressed like b, so it spans every element b strides over.
        const std::size_t span =
            (n > 0) ? static_cast<std::size_t>((n - 1) * incb + 1) : std::size_t{0};
        const ScratchBuffer work = Internal::allocate_blas_scratch(
            recorder_, b, Internal::running_solution_bytes(A.dtype) * span, operands);

        const Internal::BLASMetadata meta = Internal::BLASMetadata(op, "blas.lvl2.tbsv"_op)
                                                .flag(uplo).flag(trans).flag(diag).extent(k);

        const Core::Layout layout = a.layout;
        const int64_t lda = a.leading_dimension;
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
                        Kernels::Reference::tbsv<T>(layout, uplo, trans, diag, n, k, pA, lda, pb,
                                                    incb);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::tbsv<T>(
                            q2, layout, uplo, trans, diag, n, k, pA, lda, pb, incb,
                            static_cast<Kernels::Accumulator<T>*>(work.data()), deps2);
                    });
            });
    }
} // namespace SushiBLAS
