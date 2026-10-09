/**************************************************************************/
/* symm.cpp                                                               */
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
 * @file symm.cpp
 * @brief Defines Level3::symm and Level3::hemm, which record the BLAS SYMM and HEMM tasks.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blas/level3.hpp>

#include <vector>
#include <complex>
#include <SushiBLAS/blas_kernels/blas.hpp>
#include <SushiBLAS/blas/blas_task.hpp>
#include <SushiBLAS/blas/packed_product_scratch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/operands/task_metadata.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/matrix_operand.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_seam_kernels/symmetric_access.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm_accessed_packed.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_epilogue.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /**
         * @brief Records C = alpha * A * B + beta * C or C = alpha * B * A + beta * C for a
         *        symmetric or Hermitian A.
         * @tparam Conj Kernels::Unconjugated for SYMM, Kernels::Conjugated for HEMM.
         */
        template<typename Conj>
        TaskHandle record_symm(TaskRecorder& recorder, const char* op,
                               SushiRuntime::Graph::OpID op_id,
                               const Tensor& A, const Tensor& B, Tensor& C,
                               Core::Side side, Core::Uplo uplo,
                               BlasScalar alpha, BlasScalar beta)
        {
            Dispatch::require_flag(side, op, "side");
            Dispatch::require_flag(uplo, op, "uplo");
            Dispatch::require_dtype(B, A.dtype, op, "B");
            Dispatch::require_dtype(C, A.dtype, op, "C");
            Dispatch::require_real_scalar(alpha, A.dtype, op, "alpha");
            Dispatch::require_real_scalar(beta, A.dtype, op, "beta");

            const Dispatch::MatrixOperand a = Dispatch::describe_matrix_operand(A, op, "A");
            const Dispatch::MatrixOperand b = Dispatch::describe_matrix_operand(B, op, "B");
            const Dispatch::MatrixOperand c = Dispatch::describe_matrix_operand(C, op, "C");
            Dispatch::require_same_layout(a, c, op, "A", "C");
            Dispatch::require_same_layout(b, c, op, "B", "C");
            Dispatch::require_square(a, op, "A");

            const bool left = (side == Core::Side::Left);
            const int64_t m = c.rows;
            const int64_t n = c.cols;
            Dispatch::require_matrix_extent(a.rows, left ? m : n,
                                            left ? "rows against the rows of C"
                                                 : "rows against the columns of C",
                                            op, "A");
            Dispatch::require_matrix_extent(b.rows, m, "rows against the rows of C", op, "B");
            Dispatch::require_matrix_extent(b.cols, n, "columns against the columns of C", op, "B");

            const int64_t batch_size = c.batch_count;
            const int64_t str_a = Dispatch::batch_stride(a, batch_size, op, "A");
            const int64_t str_b = Dispatch::batch_stride(b, batch_size, op, "B");
            const int64_t str_c = Dispatch::batch_stride(c, batch_size, op, "C");
            Dispatch::require_same_batch_axes(A, a, C, c, op, "A", "C");
            Dispatch::require_same_batch_axes(B, b, C, c, op, "B", "C");

            // C is read unless beta is zero; see modules/operations/blas/README.md.
            OperandSet operands;
            operands.reads(A).reads(B);
            if (beta.is_zero())
                operands.writes(C);
            else
                operands.updates(C);

            const ScratchBuffer pack = Internal::allocate_packed_product_scratch(
                recorder, C, m, n, left ? m : n, batch_size, operands);

            const Internal::BlasMetadata meta = Internal::BlasMetadata(op, op_id)
                                                    .scalar(alpha).scalar(beta)
                                                    .flag(side).flag(uplo);

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
                    auto pA = static_cast<const T*>(read_A);
                    auto pB = static_cast<const T*>(read_B);
                    auto pC = static_cast<T*>(rw_C);
                    const T alpha_t = alpha.as<T>();
                    const T beta_t = beta.as<T>();

                    const Kernels::SymmetricAccess<Policy> at_a =
                        Kernels::symmetric_access<Policy>(layout, uplo, lda);
                    const Kernels::OperandAccess at_b =
                        Kernels::operand_access(layout, Core::Transpose::NoTrans, ldb);
                    const Kernels::OperandAccess at_c =
                        Kernels::operand_access(layout, Core::Transpose::NoTrans, ldc);
                    const Kernels::GemmIdentityEpilogue identity{};

                    return dispatcher.run(q, deps,
                        [=]
                        {
                            if (left)
                                Kernels::Reference::gemm_accessed<T>(m, n, m, alpha_t,
                                                                     pA, at_a, str_a,
                                                                     pB, at_b, str_b,
                                                                     beta_t, pC, at_c, str_c,
                                                                     batch_size, identity);
                            else
                                Kernels::Reference::gemm_accessed<T>(m, n, n, alpha_t,
                                                                     pB, at_b, str_b,
                                                                     pA, at_a, str_a,
                                                                     beta_t, pC, at_c, str_c,
                                                                     batch_size, identity);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            if (left)
                                return Kernels::SYCL::gemm_accessed<T>(q2, m, n, m, alpha_t,
                                                                       pA, at_a, str_a,
                                                                       pB, at_b, str_b,
                                                                       beta_t, pC, at_c, str_c,
                                                                       batch_size,
                                                                       pack.data(), pack.bytes(),
                                                                       deps2, identity);

                            return Kernels::SYCL::gemm_accessed<T>(q2, m, n, n, alpha_t,
                                                                   pB, at_b, str_b,
                                                                   pA, at_a, str_a,
                                                                   beta_t, pC, at_c, str_c,
                                                                   batch_size,
                                                                   pack.data(), pack.bytes(),
                                                                   deps2, identity);
                        });
                });
        }
    } // namespace

    TaskHandle Level3::symm(const Tensor& A, const Tensor& B, Tensor& C,
                            Core::Side side, Core::Uplo uplo,
                            BlasScalar alpha, BlasScalar beta)
    {
        return record_symm<Kernels::Unconjugated>(recorder_, "blas.lvl3.symm", "blas.lvl3.symm"_op,
                                                  A, B, C, side, uplo, alpha, beta);
    }

    TaskHandle Level3::hemm(const Tensor& A, const Tensor& B, Tensor& C,
                            Core::Side side, Core::Uplo uplo,
                            BlasScalar alpha, BlasScalar beta)
    {
        if (!Core::is_complex_dtype(A.dtype))
            return symm(A, B, C, side, uplo, alpha, beta);

        return record_symm<Kernels::Conjugated>(recorder_, "blas.lvl3.hemm", "blas.lvl3.hemm"_op,
                                                A, B, C, side, uplo, alpha, beta);
    }
} // namespace SushiBLAS
