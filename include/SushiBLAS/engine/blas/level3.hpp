/**************************************************************************/
/* level3.hpp                                                             */
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
 * @file level3.hpp
 * @brief Declares Level3, the matrix-matrix BLAS operations.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine/blas/scalar.hpp>

namespace SushiBLAS
{
    /**
     * @brief Records the BLAS Level 3 matrix-matrix operations through a task recorder.
     * @see include/SushiBLAS/engine/blas/README.md
     */
    class Level3
    {
        public:
            /**
             * @brief Records C = alpha * op(A) * op(B) + beta * C, batched above rank 2.
             * @param C Must not overlap A or B.
             * @param beta Exactly zero skips reading C.
             * @throws std::invalid_argument on an operand, flag or scalar the call does not accept.
             */
            TaskHandle gemm(const Tensor& A, const Tensor& B, Tensor& C,
                            Core::Transpose transA = Core::Transpose::NO_TRANS,
                            Core::Transpose transB = Core::Transpose::NO_TRANS,
                            BlasScalar alpha = 1.0, BlasScalar beta = 0.0);

            /**
             * @brief Records the solution of a triangular system, overwriting B with X.
             * @param side LEFT solves op(A) * X = alpha * B, RIGHT solves X * op(A) = alpha * B.
             * @throws std::invalid_argument on an operand, flag or scalar the call does not accept.
             */
            TaskHandle trsm(const Tensor& A, Tensor& B, Core::Side side, Core::Uplo uplo,
                            Core::Transpose trans = Core::Transpose::NO_TRANS,
                            Core::Diag diag = Core::Diag::NON_UNIT, BlasScalar alpha = 1.0);

            /**
             * @brief Records the symmetric rank-k update C = alpha * op(A) * op(A)^T + beta * C.
             * @param uplo The triangle of C that is updated.
             * @param trans NO_TRANS or TRANS; CONJ_TRANS only for a real tensor, where it is TRANS.
             * @throws std::invalid_argument on an operand, flag or scalar the call does not accept.
             */
            TaskHandle syrk(const Tensor& A, Tensor& C, Core::Uplo uplo,
                            Core::Transpose trans = Core::Transpose::NO_TRANS,
                            BlasScalar alpha = 1.0, BlasScalar beta = 0.0);

            /**
             * @brief Records the Hermitian rank-k update C = alpha * op(A) * op(A)^H + beta * C.
             * @param uplo The triangle of C that is updated; its diagonal is left real.
             * @param trans NO_TRANS or CONJ_TRANS for a complex tensor.
             * @throws std::invalid_argument on an operand or a flag the call does not accept.
             */
            TaskHandle herk(const Tensor& A, Tensor& C, Core::Uplo uplo,
                            Core::Transpose trans = Core::Transpose::NO_TRANS,
                            double alpha = 1.0, double beta = 0.0);

        protected:
            /** @brief Binds the recorder every operation records on; it must outlive this view. */
            explicit Level3(TaskRecorder& recorder) noexcept
                : recorder_(recorder)
            {

            }

            /** @brief Copies the binding. */
            Level3(const Level3&) noexcept = default;

            /** @brief Moves the binding. */
            Level3(Level3&&) noexcept = default;

            /** @brief Refused: a view cannot be re-bound. */
            Level3& operator=(const Level3&) = delete;

            /** @brief Refused: a view cannot be re-bound. */
            Level3& operator=(Level3&&) = delete;

            /** @brief Destroys the view; the recorder is not owned. */
            ~Level3() = default;

        private:
            /// @brief The recorder every operation of this level records on.
            TaskRecorder& recorder_;
    };
} // namespace SushiBLAS
