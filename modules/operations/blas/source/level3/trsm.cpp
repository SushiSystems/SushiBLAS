/**************************************************************************/
/* trsm.cpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file trsm.cpp
 * @brief Defines Level3::trsm, which records the BLAS TRSM task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blas/level3.hpp>

#include <vector>
#include <complex>
#include <cstddef>
#include <SushiBLAS/blas_kernels/blas.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/blas/blas_task.hpp>
#include <SushiBLAS/blas/shape_check.hpp>
#include <SushiBLAS/operands/scratch_lease.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/operands/blas_metadata.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/matrix_operand.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    TaskHandle Level3::trsm(const Tensor& A, Tensor& B, Core::Side side, Core::Uplo uplo,
                            Core::Transpose trans, Core::Diag diag, BLASScalar alpha)
    {
        const char* const op = "blas.lvl3.trsm";

        Dispatch::require_flag(side, op, "side");
        Dispatch::require_flag(uplo, op, "uplo");
        Dispatch::require_flag(trans, op, "trans");
        Dispatch::require_flag(diag, op, "diag");
        Dispatch::require_dtype(B, A.dtype, op, "B");
        Dispatch::require_real_scalar(alpha, A.dtype, op, "alpha");

        Dispatch::require_operand_storage(A, op, "A");
        Dispatch::require_operand_storage(B, op, "B");
        const Dispatch::TrsmGeometry geometry = Dispatch::check_trsm_shapes(A, B, side, op);
        const Dispatch::MatrixOperand& a = geometry.a;
        const Dispatch::MatrixOperand& b = geometry.b;
        const int64_t m = geometry.m;
        const int64_t n = geometry.n;
        const int64_t batch_size = geometry.batch_size;
        const int64_t str_a = geometry.stride_a;
        const int64_t str_b = geometry.stride_b;

        // B arrives as the right-hand sides and leaves as the solution X: read, then written.
        OperandSet operands;
        operands.reads(A).updates(B);
        const ScratchBuffer work = Internal::allocate_blas_scratch(
            recorder_, B,
            Internal::running_solution_bytes(A.dtype) *
                static_cast<std::size_t>(batch_size * m * n),
            operands);

        const Internal::BLASMetadata metadata = Internal::BLASMetadata(op, "blas.lvl3.trsm"_op)
                                                .scalar(alpha)
                                                .flag(side).flag(uplo).flag(trans).flag(diag);

        const Core::Layout layout = b.layout;
        const int64_t lda = a.leading_dimension;
        const int64_t ldb = b.leading_dimension;
        const void* read_A = A.data_or_null();
        void* rw_B = B.data_or_null();

        return Internal::execute_blas_task(recorder_, metadata.get(), operands, A.dtype,
            [=](auto scalar_type, const KernelPairDispatcher& dispatcher,
                const std::vector<sycl::event>& deps) -> Device::SubmittedEvents
            {
                using T = decltype(scalar_type);
                auto pA = static_cast<const T*>(read_A);
                auto pB = static_cast<T*>(rw_B);
                const T alpha_t = alpha.as<T>();

                return dispatcher.run(deps,
                    [=]
                    {
                        Kernels::Reference::trsm<T>(layout, side, uplo, trans, diag, m, n, alpha_t,
                                                    pA, lda, str_a, pB, ldb, str_b, batch_size);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> Device::SubmittedEvents
                    {
                        return Kernels::SYCL::trsm<T>(
                            q2, layout, side, uplo, trans, diag, m, n, alpha_t, pA, lda, str_a,
                            pB, ldb, str_b, batch_size,
                            static_cast<Kernels::Accumulator<T>*>(work.data()), deps2);
                    });
            });
    }
} // namespace SushiBLAS
