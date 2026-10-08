/**************************************************************************/
/* syr2k.cpp                                                              */
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
 * @file syr2k.cpp
 * @brief Defines Level3::syr2k and Level3::her2k, which record the BLAS SYR2K and HER2K tasks.
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
         * @brief Records the rank-2k update of one triangle of C from op(A) and op(B).
         * @tparam Conj Kernels::Unconjugated for SYR2K, Kernels::Conjugated for HER2K.
         * @param beta Real when Conj is Kernels::Conjugated; the caller guarantees it.
         */
        template<typename Conj>
        TaskHandle record_syr2k(TaskRecorder& recorder, const char* op,
                                SushiRuntime::Graph::OpID op_id,
                                const Tensor& A, const Tensor& B, Tensor& C,
                                Core::Uplo uplo, Core::Transpose trans,
                                BlasScalar alpha, BlasScalar beta)
        {
            Dispatch::require_flag(uplo, op, "uplo");
            Dispatch::require_flag(trans, op, "trans");
            Dispatch::require_dtype(B, A.dtype, op, "B");
            Dispatch::require_dtype(C, A.dtype, op, "C");
            Dispatch::require_real_scalar(alpha, A.dtype, op, "alpha");
            Dispatch::require_real_scalar(beta, A.dtype, op, "beta");

            const Dispatch::MatrixOperand a = Dispatch::describe_matrix_operand(A, op, "A");
            const Dispatch::MatrixOperand b = Dispatch::describe_matrix_operand(B, op, "B");
            const Dispatch::MatrixOperand c = Dispatch::describe_matrix_operand(C, op, "C");
            Dispatch::require_same_layout(a, c, op, "A", "C");
            Dispatch::require_same_layout(b, c, op, "B", "C");
            Dispatch::require_square(c, op, "C");

            const bool transposed = Core::is_transposed(trans);
            const int64_t n = c.rows;
            const int64_t k = transposed ? a.rows : a.cols;
            Dispatch::require_matrix_extent(transposed ? a.cols : a.rows, n, "rows in op(A)", op,
                                            "A");
            Dispatch::require_matrix_extent(transposed ? b.cols : b.rows, n, "rows in op(B)", op,
                                            "B");
            Dispatch::require_matrix_extent(transposed ? b.rows : b.cols, k,
                                            "columns in op(B) against the columns of op(A)", op,
                                            "B");

            const int64_t batch_size = c.batch_count;
            const int64_t str_a = Dispatch::batch_stride(a, batch_size, op, "A");
            const int64_t str_b = Dispatch::batch_stride(b, batch_size, op, "B");
            const int64_t str_c = Dispatch::batch_stride(c, batch_size, op, "C");
            Dispatch::require_same_batch_axes(A, a, C, c, op, "A", "C");
            Dispatch::require_same_batch_axes(B, b, C, c, op, "B", "C");

            Internal::require_local_tile<Kernels::SYCL::Syr2kTile>(recorder, C, op);

            // C is read and written whatever beta is: one triangle is kept; see src/README.md.
            OperandSet operands;
            operands.reads(A).reads(B).updates(C);

            const Internal::BlasMetadata meta = Internal::BlasMetadata(op, op_id)
                                                    .scalar(alpha).scalar(beta)
                                                    .flag(uplo).flag(trans);

            const Core::Layout layout = c.layout;
            const int64_t lda = a.leading_dimension;
            const int64_t ldb = b.leading_dimension;
            const int64_t ldc = c.leading_dimension;
            const void* read_A = A.data_or_null();
            const void* read_B = B.data_or_null();
            void* rw_C = C.data_or_null();

            return Internal::execute_blas_task(recorder, meta.get(), operands, A.dtype,
                [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                    sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    using T = decltype(scalar_type);
                    using Policy = Kernels::ConjugationFor<T, Conj>;
                    using Scalar = Kernels::RankUpdateScalar<T, Policy>;
                    auto pA = static_cast<const T*>(read_A);
                    auto pB = static_cast<const T*>(read_B);
                    auto pC = static_cast<T*>(rw_C);
                    const T alpha_t = alpha.as<T>();
                    const Scalar beta_s = beta.as<Scalar>();

                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::syr2k<T, Policy>(layout, uplo, trans, n, k,
                                                                 alpha_t, pA, lda, str_a,
                                                                 pB, ldb, str_b,
                                                                 beta_s, pC, ldc, str_c,
                                                                 batch_size);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::syr2k<T, Policy>(q2, layout, uplo, trans, n, k,
                                                                   alpha_t, pA, lda, str_a,
                                                                   pB, ldb, str_b,
                                                                   beta_s, pC, ldc, str_c,
                                                                   batch_size, deps2);
                        });
                });
        }
    } // namespace

    TaskHandle Level3::syr2k(const Tensor& A, const Tensor& B, Tensor& C, Core::Uplo uplo,
                             Core::Transpose trans, BlasScalar alpha, BlasScalar beta)
    {
        const char* const op = "blas.lvl3.syr2k";
        if (Core::is_complex_dtype(A.dtype))
            Dispatch::require_transpose_not(trans, Core::Transpose::ConjTrans, op, "her2k");

        return record_syr2k<Kernels::Unconjugated>(recorder_, op, "blas.lvl3.syr2k"_op,
                                                   A, B, C, uplo, trans, alpha, beta);
    }

    TaskHandle Level3::her2k(const Tensor& A, const Tensor& B, Tensor& C, Core::Uplo uplo,
                             Core::Transpose trans, BlasScalar alpha, double beta)
    {
        if (!Core::is_complex_dtype(A.dtype))
            return syr2k(A, B, C, uplo, trans, alpha, beta);

        const char* const op = "blas.lvl3.her2k";
        Dispatch::require_transpose_not(trans, Core::Transpose::Trans, op, "syr2k");

        return record_syr2k<Kernels::Conjugated>(recorder_, op, "blas.lvl3.her2k"_op,
                                                 A, B, C, uplo, trans, alpha, BlasScalar(beta));
    }
} // namespace SushiBLAS
