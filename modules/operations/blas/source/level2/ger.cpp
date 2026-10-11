/**************************************************************************/
/* ger.cpp                                                                */
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
 * @file ger.cpp
 * @brief Defines Level2::ger, Level2::geru and Level2::gerc, which record the BLAS GER, GERU
 *        and GERC tasks.
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
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    namespace
    {
        /**
         * @brief Records the rank-1 update A = alpha * x * y^T + A, or its conjugated form.
         * @tparam Conj Kernels::Unconjugated for GER and GERU, Kernels::Conjugated for GERC.
         */
        template<typename Conj>
        TaskHandle record_ger(TaskRecorder& recorder, const char* op,
                              SushiRuntime::OperationID op_id,
                              const Tensor& x, const Tensor& y, Tensor& A, BLASScalar alpha)
        {
            Dispatch::require_dtype(x, A.dtype, op, "x");
            Dispatch::require_dtype(y, A.dtype, op, "y");
            Dispatch::require_real_scalar(alpha, A.dtype, op, "alpha");

            const Dispatch::MatrixOperand a = Dispatch::describe_matrix_operand(A, op, "A");
            Dispatch::require_unbatched(a, op, "A");
            const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");
            const Dispatch::VectorOperand vy = Dispatch::describe_vector_operand(y, op, "y");
            Dispatch::require_vector_length(vx, a.rows, "the rows of A", op, "x");
            Dispatch::require_vector_length(vy, a.cols, "the columns of A", op, "y");

            OperandSet operands;
            operands.reads(x).reads(y).updates(A);

            const Internal::BLASMetadata meta = Internal::BLASMetadata(op, op_id).scalar(alpha);

            const Core::Layout layout = a.layout;
            const int64_t m = a.rows;
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
                            Kernels::Reference::ger<T, Policy>(layout, m, n, alpha_t, px, incx,
                                                               py, incy, pA, lda);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::ger<T, Policy>(q2, layout, m, n, alpha_t, px,
                                                                 incx, py, incy, pA, lda, deps2);
                        });
                });
        }
    } // namespace

    TaskHandle Level2::ger(const Tensor& x, const Tensor& y, Tensor& A, double alpha)
    {
        const char* const op = "blas.lvl2.ger";

        Dispatch::require_real_dtype(A, op, "A", "geru or gerc");
        return record_ger<Kernels::Unconjugated>(recorder_, op, "blas.lvl2.ger"_op, x, y, A,
                                                 BLASScalar(alpha));
    }

    TaskHandle Level2::geru(const Tensor& x, const Tensor& y, Tensor& A, BLASScalar alpha)
    {
        const char* const op = "blas.lvl2.geru";

        if (!Core::is_complex_dtype(A.dtype))
        {
            Dispatch::require_real_scalar(alpha, A.dtype, op, "alpha");
            return ger(x, y, A, alpha.real());
        }

        return record_ger<Kernels::Unconjugated>(recorder_, op, "blas.lvl2.geru"_op, x, y, A,
                                                 alpha);
    }

    TaskHandle Level2::gerc(const Tensor& x, const Tensor& y, Tensor& A, BLASScalar alpha)
    {
        const char* const op = "blas.lvl2.gerc";

        if (!Core::is_complex_dtype(A.dtype))
        {
            Dispatch::require_real_scalar(alpha, A.dtype, op, "alpha");
            return ger(x, y, A, alpha.real());
        }

        return record_ger<Kernels::Conjugated>(recorder_, op, "blas.lvl2.gerc"_op, x, y, A,
                                               alpha);
    }
} // namespace SushiBLAS
