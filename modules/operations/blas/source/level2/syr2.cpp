/**************************************************************************/
/* syr2.cpp                                                               */
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
 * @file syr2.cpp
 * @brief Defines Level2::syr2 and Level2::her2, which record the BLAS SYR2 and HER2 tasks.
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
         * @brief Records the rank-2 update A = alpha * x * y^T + alpha * y * x^T + A, or its
         *        Hermitian form.
         * @tparam Conj Kernels::Unconjugated for SYR2, Kernels::Conjugated for HER2.
         */
        template<typename Conj>
        TaskHandle record_syr2(TaskRecorder& recorder, const char* op,
                               SushiRuntime::Graph::OpID op_id,
                               const Tensor& x, const Tensor& y, Tensor& A,
                               Core::Uplo uplo, BLASScalar alpha)
        {
            Dispatch::require_flag(uplo, op, "uplo");
            Dispatch::require_dtype(x, A.dtype, op, "x");
            Dispatch::require_dtype(y, A.dtype, op, "y");
            Dispatch::require_real_scalar(alpha, A.dtype, op, "alpha");

            const Dispatch::MatrixOperand a = Dispatch::describe_matrix_operand(A, op, "A");
            Dispatch::require_unbatched(a, op, "A");
            Dispatch::require_square(a, op, "A");
            const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");
            const Dispatch::VectorOperand vy = Dispatch::describe_vector_operand(y, op, "y");
            Dispatch::require_vector_length(vx, a.cols, "the order of A", op, "x");
            Dispatch::require_vector_length(vy, a.cols, "the order of A", op, "y");

            OperandSet operands;
            operands.reads(x).reads(y).updates(A);

            const Internal::BLASMetadata meta =
                Internal::BLASMetadata(op, op_id).scalar(alpha).flag(uplo);

            const Core::Layout layout = a.layout;
            const int64_t n = a.cols;
            const int64_t lda = a.leading_dimension;
            const int64_t incx = vx.increment;
            const int64_t incy = vy.increment;
            const void* read_x = x.data_or_null();
            const void* read_y = y.data_or_null();
            void* rw_A = A.data_or_null();

            return Internal::execute_blas_task(recorder, meta.get(), operands, A.dtype,
                [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                    sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    using T = decltype(scalar_type);
                    using Policy = Kernels::ConjugationFor<T, Conj>;
                    auto px = static_cast<const T*>(read_x);
                    auto py = static_cast<const T*>(read_y);
                    auto pA = static_cast<T*>(rw_A);
                    const T alpha_t = alpha.as<T>();

                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::syr2<T, Policy>(layout, uplo, n, alpha_t, px, incx,
                                                                py, incy, pA, lda);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::syr2<T, Policy>(q2, layout, uplo, n, alpha_t, px,
                                                                  incx, py, incy, pA, lda, deps2);
                        });
                });
        }
    } // namespace

    TaskHandle Level2::syr2(const Tensor& x, const Tensor& y, Tensor& A, Core::Uplo uplo,
                            BLASScalar alpha)
    {
        return record_syr2<Kernels::Unconjugated>(recorder_, "blas.lvl2.syr2", "blas.lvl2.syr2"_op,
                                                  x, y, A, uplo, alpha);
    }

    TaskHandle Level2::her2(const Tensor& x, const Tensor& y, Tensor& A, Core::Uplo uplo,
                            BLASScalar alpha)
    {
        if (!Core::is_complex_dtype(A.dtype))
            return syr2(x, y, A, uplo, alpha);

        return record_syr2<Kernels::Conjugated>(recorder_, "blas.lvl2.her2", "blas.lvl2.her2"_op,
                                                x, y, A, uplo, alpha);
    }
} // namespace SushiBLAS
