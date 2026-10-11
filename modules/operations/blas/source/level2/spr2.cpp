/**************************************************************************/
/* spr2.cpp                                                               */
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
 * @file spr2.cpp
 * @brief Defines Level2::spr2 and Level2::hpr2, which record the BLAS SPR2 and HPR2 tasks.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blas/level2.hpp>

#include <vector>
#include <complex>
#include <SushiBLAS/blas_kernels/blas.hpp>
#include <SushiBLAS/blas/blas_task.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/blas/packed_operand.hpp>
#include <SushiBLAS/operands/task_metadata.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiRuntime/task/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /**
         * @brief Records the rank-2 update of a packed symmetric or Hermitian A.
         * @tparam Conj Kernels::Unconjugated for SPR2, Kernels::Conjugated for HPR2.
         */
        template<typename Conj>
        TaskHandle record_spr2(TaskRecorder& recorder, const char* op,
                               SushiRuntime::Graph::OpID op_id,
                               const Tensor& x, const Tensor& y, Tensor& AP,
                               Core::Uplo uplo, BlasScalar alpha)
        {
            Dispatch::require_flag(uplo, op, "uplo");
            Dispatch::require_dtype(x, AP.dtype, op, "x");
            Dispatch::require_dtype(y, AP.dtype, op, "y");
            Dispatch::require_real_scalar(alpha, AP.dtype, op, "alpha");

            const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");
            const Dispatch::VectorOperand vy = Dispatch::describe_vector_operand(y, op, "y");
            const Dispatch::PackedOperand ap = Dispatch::describe_packed_operand(AP, op, "AP");
            Dispatch::require_vector_length(vx, ap.order, "the order of AP", op, "x");
            Dispatch::require_vector_length(vy, ap.order, "the order of AP", op, "y");

            OperandSet operands;
            operands.reads(x).reads(y).updates(AP);

            const Internal::BlasMetadata meta =
                Internal::BlasMetadata(op, op_id).scalar(alpha).flag(uplo);

            const Core::Layout layout = ap.layout;
            const int64_t n = ap.order;
            const int64_t incx = vx.increment;
            const int64_t incy = vy.increment;
            const void* read_x = x.data_or_null();
            const void* read_y = y.data_or_null();
            void* rw_AP = AP.data_or_null();

            return Internal::execute_blas_task(recorder, meta.get(), operands, AP.dtype,
                [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                    sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    using T = decltype(scalar_type);
                    using Policy = Kernels::ConjugationFor<T, Conj>;
                    auto px = static_cast<const T*>(read_x);
                    auto py = static_cast<const T*>(read_y);
                    auto pAP = static_cast<T*>(rw_AP);
                    const T alpha_t = alpha.as<T>();

                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::spr2<T, Policy>(layout, uplo, n, alpha_t, px, incx,
                                                                py, incy, pAP);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::spr2<T, Policy>(q2, layout, uplo, n, alpha_t, px,
                                                                  incx, py, incy, pAP, deps2);
                        });
                });
        }
    } // namespace

    TaskHandle Level2::spr2(const Tensor& x, const Tensor& y, Tensor& AP, Core::Uplo uplo,
                            BlasScalar alpha)
    {
        return record_spr2<Kernels::Unconjugated>(recorder_, "blas.lvl2.spr2", "blas.lvl2.spr2"_op,
                                                  x, y, AP, uplo, alpha);
    }

    TaskHandle Level2::hpr2(const Tensor& x, const Tensor& y, Tensor& AP, Core::Uplo uplo,
                            BlasScalar alpha)
    {
        if (!Core::is_complex_dtype(AP.dtype))
            return spr2(x, y, AP, uplo, alpha);

        return record_spr2<Kernels::Conjugated>(recorder_, "blas.lvl2.hpr2", "blas.lvl2.hpr2"_op,
                                                x, y, AP, uplo, alpha);
    }
} // namespace SushiBLAS
