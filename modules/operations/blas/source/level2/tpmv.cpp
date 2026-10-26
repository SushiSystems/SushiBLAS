/**************************************************************************/
/* tpmv.cpp                                                               */
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
 * @file tpmv.cpp
 * @brief Defines Level2::tpmv, which records the BLAS TPMV task.
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
#include <SushiBLAS/blas/packed_operand.hpp>
#include <SushiBLAS/operands/blas_metadata.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    TaskHandle Level2::tpmv(const Tensor& AP, Tensor& x, Core::Uplo uplo, Core::Transpose trans,
                            Core::Diag diag)
    {
        const char* const op = "blas.lvl2.tpmv";

        Dispatch::require_flag(uplo, op, "uplo");
        Dispatch::require_flag(trans, op, "trans");
        Dispatch::require_flag(diag, op, "diag");
        Dispatch::require_dtype(x, AP.dtype, op, "x");

        const Dispatch::PackedOperand ap = Dispatch::describe_packed_operand(AP, op, "AP");
        const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");
        Dispatch::require_vector_length(vx, ap.order, "the order of AP", op, "x");

        const int64_t n = ap.order;

        // The SYCL kernel reads the old x from a snapshot while it writes the new one.
        OperandSet operands;
        operands.reads(AP).updates(x);
        const ScratchBuffer snapshot = Internal::allocate_blas_scratch(
            recorder_, x, Core::bytes_per_element(AP.dtype) * static_cast<std::size_t>(n),
            operands);

        const Internal::BLASMetadata metadata =
            Internal::BLASMetadata(op, "blas.lvl2.tpmv"_op).flag(uplo).flag(trans).flag(diag);

        const Core::Layout layout = ap.layout;
        const int64_t incx = vx.increment;
        const void* read_AP = AP.data_or_null();
        void* rw_x = x.data_or_null();

        return Internal::execute_blas_task(recorder_, metadata.get(), operands, AP.dtype,
            [=](auto scalar_type, const KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> Device::SubmittedEvents
            {
                using T = decltype(scalar_type);
                auto pAP = static_cast<const T*>(read_AP);
                auto px = static_cast<T*>(rw_x);
                auto work = static_cast<T*>(snapshot.data());

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::tpmv<T>(layout, uplo, trans, diag, n, pAP, px, incx);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> Device::SubmittedEvents
                    {
                        return Kernels::SYCL::tpmv<T>(q2, layout, uplo, trans, diag, n, pAP, px,
                                                      incx, work, deps2);
                    });
            });
    }
} // namespace SushiBLAS
