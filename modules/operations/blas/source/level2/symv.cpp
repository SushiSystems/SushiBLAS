/**************************************************************************/
/* symv.cpp                                                               */
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
 * @file symv.cpp
 * @brief Defines Level2::symv and Level2::hemv, which record the BLAS SYMV and HEMV tasks.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blas/level2.hpp>

#include <vector>
#include <complex>
#include <SushiBLAS/blas_kernels/blas.hpp>
#include <SushiBLAS/blas/blas_task.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/operands/task_metadata.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/matrix_operand.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiRuntime/task/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /**
         * @brief Records y = alpha * A * x + beta * y for a symmetric or Hermitian A.
         * @tparam Conj Kernels::Unconjugated for SYMV, Kernels::Conjugated for HEMV.
         */
        template<typename Conj>
        TaskHandle record_symv(TaskRecorder& recorder, const char* op,
                               SushiRuntime::Graph::OpID op_id,
                               const Tensor& A, const Tensor& x, Tensor& y,
                               Core::Uplo uplo, BLASScalar alpha, BLASScalar beta)
        {
            Dispatch::require_flag(uplo, op, "uplo");
            Dispatch::require_dtype(x, A.dtype, op, "x");
            Dispatch::require_dtype(y, A.dtype, op, "y");
            Dispatch::require_real_scalar(alpha, A.dtype, op, "alpha");
            Dispatch::require_real_scalar(beta, A.dtype, op, "beta");

            const Dispatch::MatrixOperand a = Dispatch::describe_matrix_operand(A, op, "A");
            Dispatch::require_unbatched(a, op, "A");
            Dispatch::require_square(a, op, "A");
            const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");
            const Dispatch::VectorOperand vy = Dispatch::describe_vector_operand(y, op, "y");
            Dispatch::require_vector_length(vx, a.cols, "the columns of A", op, "x");
            Dispatch::require_vector_length(vy, a.rows, "the rows of A", op, "y");

            OperandSet operands;
            operands.reads(A).reads(x);
            if (beta.is_zero())
                operands.writes(y);
            else
                operands.updates(y);

            const Internal::BLASMetadata meta =
                Internal::BLASMetadata(op, op_id).scalar(alpha).scalar(beta).flag(uplo);

            const Core::Layout layout = a.layout;
            const int64_t n = a.rows;
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
                            Kernels::Reference::symv<T, Policy>(layout, uplo, n, alpha_t, pA, lda,
                                                                px, incx, beta_t, py, incy);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::symv<T, Policy>(q2, layout, uplo, n, alpha_t,
                                                                  pA, lda, px, incx, beta_t, py,
                                                                  incy, deps2);
                        });
                });
        }
    } // namespace

    TaskHandle Level2::symv(const Tensor& A, const Tensor& x, Tensor& y, Core::Uplo uplo,
                            BLASScalar alpha, BLASScalar beta)
    {
        return record_symv<Kernels::Unconjugated>(recorder_, "blas.lvl2.symv", "blas.lvl2.symv"_op,
                                                  A, x, y, uplo, alpha, beta);
    }

    TaskHandle Level2::hemv(const Tensor& A, const Tensor& x, Tensor& y, Core::Uplo uplo,
                            BLASScalar alpha, BLASScalar beta)
    {
        if (!Core::is_complex_dtype(A.dtype))
            return symv(A, x, y, uplo, alpha, beta);

        return record_symv<Kernels::Conjugated>(recorder_, "blas.lvl2.hemv", "blas.lvl2.hemv"_op,
                                                A, x, y, uplo, alpha, beta);
    }
} // namespace SushiBLAS
