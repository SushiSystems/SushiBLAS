/**************************************************************************/
/* sbmv.cpp                                                               */
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
 * @file sbmv.cpp
 * @brief Defines Level2::sbmv and Level2::hbmv, which record the BLAS SBMV and HBMV tasks.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blas/level2.hpp>

#include <vector>
#include <complex>
#include <SushiBLAS/blas_kernels/blas.hpp>
#include <SushiBLAS/operands/utils.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/blas/band_operand.hpp>
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
         * @brief Records y = alpha * A * x + beta * y for a symmetric or Hermitian band A.
         * @tparam Conj Kernels::Unconjugated for SBMV, Kernels::Conjugated for HBMV.
         */
        template<typename Conj>
        TaskHandle record_sbmv(TaskRecorder& recorder, const char* op,
                               SushiRuntime::Graph::OpID op_id,
                               const Tensor& A, const Tensor& x, Tensor& y,
                               Core::Uplo uplo, int64_t k, BlasScalar alpha, BlasScalar beta)
        {
            Dispatch::require_flag(uplo, op, "uplo");
            Dispatch::require_dtype(x, A.dtype, op, "x");
            Dispatch::require_dtype(y, A.dtype, op, "y");
            Dispatch::require_real_scalar(alpha, A.dtype, op, "alpha");
            Dispatch::require_real_scalar(beta, A.dtype, op, "beta");

            const Dispatch::BandOperand a =
                Dispatch::describe_triangle_band_operand(A, uplo, k, op, "A");
            const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");
            const Dispatch::VectorOperand vy = Dispatch::describe_vector_operand(y, op, "y");
            Dispatch::require_vector_length(vx, a.lines, "the order of A", op, "x");
            Dispatch::require_vector_length(vy, a.lines, "the order of A", op, "y");

            OperandSet operands;
            operands.reads(A).reads(x);
            if (beta.is_zero())
                operands.writes(y);
            else
                operands.updates(y);

            const Internal::BlasMetadata meta = Internal::BlasMetadata(op, op_id)
                                                    .scalar(alpha).scalar(beta)
                                                    .flag(uplo).extent(k);

            const Core::Layout layout = a.layout;
            const int64_t n = a.lines;
            const int64_t lda = a.leading_dimension;
            const int64_t incx = vx.increment;
            const int64_t incy = vy.increment;
            const void* read_A = A.data_or_null();
            const void* read_x = x.data_or_null();
            void* rw_y = y.data_or_null();

            return Internal::execute_blas_task(recorder, meta.get(), operands, A.dtype,
                [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                    sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    using T = decltype(scalar_type);
                    using Policy = Kernels::ConjugationFor<T, Conj>;
                    auto pA = static_cast<const T*>(read_A);
                    auto px = static_cast<const T*>(read_x);
                    auto py = static_cast<T*>(rw_y);
                    const T alpha_t = alpha.as<T>();
                    const T beta_t = beta.as<T>();

                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::sbmv<T, Policy>(layout, uplo, n, k, alpha_t, pA,
                                                                lda, px, incx, beta_t, py, incy);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::sbmv<T, Policy>(q2, layout, uplo, n, k, alpha_t,
                                                                  pA, lda, px, incx, beta_t, py,
                                                                  incy, deps2);
                        });
                });
        }
    } // namespace

    TaskHandle Level2::sbmv(const Tensor& A, const Tensor& x, Tensor& y, Core::Uplo uplo,
                            int64_t k, BlasScalar alpha, BlasScalar beta)
    {
        return record_sbmv<Kernels::Unconjugated>(recorder_, "blas.lvl2.sbmv", "blas.lvl2.sbmv"_op,
                                                  A, x, y, uplo, k, alpha, beta);
    }

    TaskHandle Level2::hbmv(const Tensor& A, const Tensor& x, Tensor& y, Core::Uplo uplo,
                            int64_t k, BlasScalar alpha, BlasScalar beta)
    {
        if (!Core::is_complex_dtype(A.dtype))
            return sbmv(A, x, y, uplo, k, alpha, beta);

        return record_sbmv<Kernels::Conjugated>(recorder_, "blas.lvl2.hbmv", "blas.lvl2.hbmv"_op,
                                                A, x, y, uplo, k, alpha, beta);
    }
} // namespace SushiBLAS
