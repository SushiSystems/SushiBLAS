/**************************************************************************/
/* spr.cpp                                                                */
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
 * @file spr.cpp
 * @brief Defines Level2::spr and Level2::hpr, which record the BLAS SPR and HPR tasks.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blas/level2.hpp>

#include <vector>
#include <complex>
#include <SushiBLAS/blas_kernels/blas.hpp>
#include <SushiBLAS/operands/utils.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/blas/packed_operand.hpp>
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
         * @brief Records the rank-1 update of a packed symmetric or Hermitian A.
         * @tparam Conj Kernels::Unconjugated for SPR, Kernels::Conjugated for HPR.
         */
        template<typename Conj>
        TaskHandle record_spr(TaskRecorder& recorder, const char* op,
                              SushiRuntime::Graph::OpID op_id,
                              const Tensor& x, Tensor& AP, Core::Uplo uplo, BlasScalar alpha)
        {
            Dispatch::require_flag(uplo, op, "uplo");
            Dispatch::require_dtype(x, AP.dtype, op, "x");
            Dispatch::require_real_scalar(alpha, AP.dtype, op, "alpha");

            const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");
            const Dispatch::PackedOperand ap = Dispatch::describe_packed_operand(AP, op, "AP");
            Dispatch::require_vector_length(vx, ap.order, "the order of AP", op, "x");

            OperandSet operands;
            operands.reads(x).updates(AP);

            const Internal::BlasMetadata meta =
                Internal::BlasMetadata(op, op_id).scalar(alpha).flag(uplo);

            const Core::Layout layout = ap.layout;
            const int64_t n = ap.order;
            const int64_t incx = vx.increment;
            const void* read_x = x.data_or_null();
            void* rw_AP = AP.data_or_null();

            return Internal::execute_blas_task(recorder, meta.get(), operands, AP.dtype,
                [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                    sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    using T = decltype(scalar_type);
                    using Policy = Kernels::ConjugationFor<T, Conj>;
                    using Scalar = Kernels::RankUpdateScalar<T, Policy>;
                    auto px = static_cast<const T*>(read_x);
                    auto pAP = static_cast<T*>(rw_AP);
                    const Scalar alpha_s = alpha.as<Scalar>();

                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::spr<T, Policy>(layout, uplo, n, alpha_s, px, incx,
                                                               pAP);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::spr<T, Policy>(q2, layout, uplo, n, alpha_s, px,
                                                                 incx, pAP, deps2);
                        });
                });
        }
    } // namespace

    TaskHandle Level2::spr(const Tensor& x, Tensor& AP, Core::Uplo uplo, BlasScalar alpha)
    {
        return record_spr<Kernels::Unconjugated>(recorder_, "blas.lvl2.spr", "blas.lvl2.spr"_op,
                                                 x, AP, uplo, alpha);
    }

    TaskHandle Level2::hpr(const Tensor& x, Tensor& AP, Core::Uplo uplo, double alpha)
    {
        if (!Core::is_complex_dtype(AP.dtype))
            return spr(x, AP, uplo, alpha);

        return record_spr<Kernels::Conjugated>(recorder_, "blas.lvl2.hpr", "blas.lvl2.hpr"_op,
                                               x, AP, uplo, BlasScalar(alpha));
    }
} // namespace SushiBLAS
