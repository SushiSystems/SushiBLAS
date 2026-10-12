/**************************************************************************/
/* syr.cpp                                                                */
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
 * @file syr.cpp
 * @brief Defines Level2::syr and Level2::her, which record the BLAS SYR and HER tasks.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blas/level2.hpp>

#include <vector>
#include <complex>
#include <SushiBLAS/blas_kernels/blas.hpp>
#include <SushiBLAS/blas/blas_task.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/operands/blas_metadata.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/matrix_operand.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    namespace
    {
        /**
         * @brief Records the rank-1 update A = alpha * x * x^T + A, or its Hermitian form.
         * @tparam Conj Kernels::Unconjugated for SYR, Kernels::Conjugated for HER.
         */
        template<typename Conj>
        TaskHandle record_syr(TaskRecorder& recorder, const char* op,
                              SushiRuntime::OperationID op_id,
                              const Tensor& x, Tensor& A, Core::Uplo uplo, BLASScalar alpha)
        {
            Dispatch::require_flag(uplo, op, "uplo");
            Dispatch::require_dtype(x, A.dtype, op, "x");
            Dispatch::require_real_scalar(alpha, A.dtype, op, "alpha");

            const Dispatch::MatrixOperand a = Dispatch::describe_matrix_operand(A, op, "A");
            Dispatch::require_unbatched(a, op, "A");
            Dispatch::require_square(a, op, "A");
            const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");
            Dispatch::require_vector_length(vx, a.cols, "the order of A", op, "x");

            OperandSet operands;
            operands.reads(x).updates(A);

            const Internal::BLASMetadata metadata =
                Internal::BLASMetadata(op, op_id).scalar(alpha).flag(uplo);

            const Core::Layout layout = a.layout;
            const int64_t n = a.cols;
            const int64_t lda = a.leading_dimension;
            const int64_t incx = vx.increment;
            const void* read_x = x.data_or_null();
            void* rw_A = A.data_or_null();

            return Internal::execute_blas_task(recorder, metadata.get(), operands, A.dtype,
                [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                    sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    using T = decltype(scalar_type);
                    using Policy = Kernels::ConjugationFor<T, Conj>;
                    using Scalar = Kernels::RankUpdateScalar<T, Policy>;
                    auto px = static_cast<const T*>(read_x);
                    auto pA = static_cast<T*>(rw_A);
                    const Scalar alpha_s = alpha.as<Scalar>();

                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::syr<T, Policy>(layout, uplo, n, alpha_s, px, incx,
                                                               pA, lda);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::syr<T, Policy>(q2, layout, uplo, n, alpha_s, px,
                                                                 incx, pA, lda, deps2);
                        });
                });
        }
    } // namespace

    TaskHandle Level2::syr(const Tensor& x, Tensor& A, Core::Uplo uplo, BLASScalar alpha)
    {
        return record_syr<Kernels::Unconjugated>(recorder_, "blas.lvl2.syr", "blas.lvl2.syr"_op,
                                                 x, A, uplo, alpha);
    }

    TaskHandle Level2::her(const Tensor& x, Tensor& A, Core::Uplo uplo, double alpha)
    {
        if (!Core::is_complex_dtype(A.dtype))
            return syr(x, A, uplo, alpha);

        return record_syr<Kernels::Conjugated>(recorder_, "blas.lvl2.her", "blas.lvl2.her"_op,
                                               x, A, uplo, BLASScalar(alpha));
    }
} // namespace SushiBLAS
