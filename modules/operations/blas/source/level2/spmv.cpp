/**************************************************************************/
/* spmv.cpp                                                               */
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
 * @file spmv.cpp
 * @brief Defines Level2::spmv and Level2::hpmv, which record the BLAS SPMV and HPMV tasks.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blas/level2.hpp>

#include <vector>
#include <complex>
#include <SushiBLAS/blas_kernels/blas.hpp>
#include <SushiBLAS/blas/blas_task.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/blas/packed_operand.hpp>
#include <SushiBLAS/operands/blas_metadata.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    namespace
    {
        /**
         * @brief Records y = alpha * A * x + beta * y for a packed symmetric or Hermitian A.
         * @tparam Conj Kernels::Unconjugated for SPMV, Kernels::Conjugated for HPMV.
         */
        template<typename Conj>
        TaskHandle record_spmv(TaskRecorder& recorder, const char* op,
                               SushiRuntime::OperationID op_id,
                               const Tensor& AP, const Tensor& x, Tensor& y,
                               Core::Uplo uplo, BLASScalar alpha, BLASScalar beta)
        {
            Dispatch::require_flag(uplo, op, "uplo");
            Dispatch::require_dtype(x, AP.dtype, op, "x");
            Dispatch::require_dtype(y, AP.dtype, op, "y");
            Dispatch::require_real_scalar(alpha, AP.dtype, op, "alpha");
            Dispatch::require_real_scalar(beta, AP.dtype, op, "beta");

            const Dispatch::PackedOperand ap = Dispatch::describe_packed_operand(AP, op, "AP");
            const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");
            const Dispatch::VectorOperand vy = Dispatch::describe_vector_operand(y, op, "y");
            Dispatch::require_vector_length(vx, ap.order, "the order of AP", op, "x");
            Dispatch::require_vector_length(vy, ap.order, "the order of AP", op, "y");

            OperandSet operands;
            operands.reads(AP).reads(x);
            if (beta.is_zero())
                operands.writes(y);
            else
                operands.updates(y);

            const Internal::BLASMetadata metadata =
                Internal::BLASMetadata(op, op_id).scalar(alpha).scalar(beta).flag(uplo);

            const Core::Layout layout = ap.layout;
            const int64_t n = ap.order;
            const int64_t incx = vx.increment;
            const int64_t incy = vy.increment;
            const void* read_AP = AP.data_or_null();
            const void* read_x = x.data_or_null();
            void* rw_y = y.data_or_null();

            return Internal::execute_blas_task(recorder, metadata.get(), operands, AP.dtype,
                [=](auto scalar_type, const KernelPairDispatcher& dispatcher,
                    sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    using T = decltype(scalar_type);
                    using Policy = Kernels::ConjugationFor<T, Conj>;
                    auto pAP = static_cast<const T*>(read_AP);
                    auto px = static_cast<const T*>(read_x);
                    auto py = static_cast<T*>(rw_y);
                    const T alpha_t = alpha.as<T>();
                    const T beta_t = beta.as<T>();

                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::spmv<T, Policy>(layout, uplo, n, alpha_t, pAP, px,
                                                                incx, beta_t, py, incy);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::spmv<T, Policy>(q2, layout, uplo, n, alpha_t,
                                                                  pAP, px, incx, beta_t, py, incy,
                                                                  deps2);
                        });
                });
        }
    } // namespace

    TaskHandle Level2::spmv(const Tensor& AP, const Tensor& x, Tensor& y, Core::Uplo uplo,
                            BLASScalar alpha, BLASScalar beta)
    {
        return record_spmv<Kernels::Unconjugated>(recorder_, "blas.lvl2.spmv", "blas.lvl2.spmv"_op,
                                                  AP, x, y, uplo, alpha, beta);
    }

    TaskHandle Level2::hpmv(const Tensor& AP, const Tensor& x, Tensor& y, Core::Uplo uplo,
                            BLASScalar alpha, BLASScalar beta)
    {
        if (!Core::is_complex_dtype(AP.dtype))
            return spmv(AP, x, y, uplo, alpha, beta);

        return record_spmv<Kernels::Conjugated>(recorder_, "blas.lvl2.hpmv", "blas.lvl2.hpmv"_op,
                                                AP, x, y, uplo, alpha, beta);
    }
} // namespace SushiBLAS
