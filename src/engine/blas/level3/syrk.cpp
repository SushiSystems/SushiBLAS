/**************************************************************************/
/* syrk.cpp                                                               */
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
 * @file syrk.cpp
 * @brief Defines Level3::syrk and Level3::herk, which record the BLAS SYRK and HERK tasks.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/blas/level3.hpp>

#include <vector>
#include <complex>
#include <kernels/blas.hpp>
#include <SushiBLAS/engine/blas/utils.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/blas/task_metadata.hpp>
#include <SushiBLAS/engine/blas/argument_check.hpp>
#include <SushiBLAS/engine/blas/matrix_operand.hpp>
#include <SushiBLAS/kernels/blas/conjugation.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /**
         * @brief Records C = alpha * op(A) * op(A)^T + beta * C, or its Hermitian form.
         * @tparam Conj Kernels::Unconjugated for SYRK, Kernels::Conjugated for HERK.
         */
        template<typename Conj>
        TaskHandle record_syrk(TaskRecorder& recorder, const char* op,
                               SushiRuntime::Graph::OpID op_id,
                               const Tensor& A, Tensor& C,
                               Core::Uplo uplo, Core::Transpose trans,
                               BlasScalar alpha, BlasScalar beta)
        {
            Dispatch::require_flag(uplo, op, "uplo");
            Dispatch::require_flag(trans, op, "trans");
            Dispatch::require_dtype(C, A.dtype, op, "C");
            Dispatch::require_real_scalar(alpha, A.dtype, op, "alpha");
            Dispatch::require_real_scalar(beta, A.dtype, op, "beta");

            const Dispatch::MatrixOperand a = Dispatch::describe_matrix_operand(A, op, "A");
            const Dispatch::MatrixOperand c = Dispatch::describe_matrix_operand(C, op, "C");
            Dispatch::require_same_layout(a, c, op, "A", "C");
            Dispatch::require_square(c, op, "C");

            const bool transposed = Core::is_transposed(trans);
            const int64_t n = c.rows;
            const int64_t k = transposed ? a.rows : a.cols;
            Dispatch::require_matrix_extent(transposed ? a.cols : a.rows, n, "rows in op(A)", op,
                                            "A");

            const int64_t batch_size = c.batch_count;
            const int64_t str_a = Dispatch::batch_stride(a, batch_size, op, "A");
            const int64_t str_c = Dispatch::batch_stride(c, batch_size, op, "C");
            Dispatch::require_same_batch_axes(A, a, C, c, op, "A", "C");

            // C is read-modify-write whatever beta is; see src/README.md.
            OperandSet operands;
            operands.reads(A).updates(C);

            const ScratchBuffer pack = Internal::allocate_packed_product_scratch(
                recorder, C, n, n, k, batch_size, operands);

            const Internal::BlasMetadata meta = Internal::BlasMetadata(op, op_id)
                                                    .scalar(alpha).scalar(beta)
                                                    .flag(uplo).flag(trans);

            const Core::Layout layout = c.layout;
            const int64_t lda = a.leading_dimension;
            const int64_t ldc = c.leading_dimension;
            const void* read_A = A.data_or_null();
            void* rw_C = C.data_or_null();

            return Internal::execute_blas_task(recorder, meta.get(), operands, A.dtype,
                [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                    sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    using T = decltype(scalar_type);
                    using Policy = Kernels::ConjugationFor<T, Conj>;
                    using Scalar = Kernels::RankUpdateScalar<T, Policy>;
                    auto pA = static_cast<const T*>(read_A);
                    auto pC = static_cast<T*>(rw_C);
                    const Scalar alpha_s = alpha.as<Scalar>();
                    const Scalar beta_s = beta.as<Scalar>();

                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::syrk<T, Policy>(layout, uplo, trans, n, k, alpha_s,
                                                                pA, lda, str_a, beta_s, pC, ldc,
                                                                str_c, batch_size);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::syrk<T, Policy>(q2, layout, uplo, trans, n, k,
                                                                  alpha_s, pA, lda, str_a, beta_s,
                                                                  pC, ldc, str_c, batch_size,
                                                                  pack.data(), pack.bytes(),
                                                                  deps2);
                        });
                });
        }
    } // namespace

    TaskHandle Level3::syrk(const Tensor& A, Tensor& C, Core::Uplo uplo, Core::Transpose trans,
                            BlasScalar alpha, BlasScalar beta)
    {
        const char* const op = "blas.lvl3.syrk";
        if (Core::is_complex_dtype(A.dtype))
            Dispatch::require_transpose_not(trans, Core::Transpose::ConjTrans, op, "herk");

        return record_syrk<Kernels::Unconjugated>(recorder_, op, "blas.lvl3.syrk"_op,
                                                  A, C, uplo, trans, alpha, beta);
    }

    TaskHandle Level3::herk(const Tensor& A, Tensor& C, Core::Uplo uplo, Core::Transpose trans,
                            double alpha, double beta)
    {
        if (!Core::is_complex_dtype(A.dtype))
            return syrk(A, C, uplo, trans, alpha, beta);

        const char* const op = "blas.lvl3.herk";
        Dispatch::require_transpose_not(trans, Core::Transpose::Trans, op, "syrk");

        return record_syrk<Kernels::Conjugated>(recorder_, op, "blas.lvl3.herk"_op,
                                                A, C, uplo, trans,
                                                BlasScalar(alpha), BlasScalar(beta));
    }
} // namespace SushiBLAS
