/**************************************************************************/
/* level2.hpp                                                             */
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
 * @file level2.hpp
 * @brief Declares Level2, the matrix-vector BLAS operations.
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
     * @brief Records the BLAS Level 2 matrix-vector operations through a task recorder.
     * @see include/SushiBLAS/engine/blas/README.md
     */
    class Level2
    {
        public:
            /**
             * @brief Records y = alpha * op(A) * x + beta * y.
             * @param beta Exactly zero skips reading y.
             * @throws std::invalid_argument on an operand, flag or scalar the call does not accept.
             */
            TaskHandle gemv(const Tensor& A, const Tensor& x, Tensor& y,
                            Core::Transpose trans = Core::Transpose::NO_TRANS,
                            BlasScalar alpha = 1.0, BlasScalar beta = 0.0);

            /**
             * @brief Records the rank-1 update A = alpha * x * y^T + A of a real matrix.
             * @throws std::invalid_argument on a complex operand, or one the call does not accept.
             */
            TaskHandle ger(const Tensor& x, const Tensor& y, Tensor& A, double alpha = 1.0);

            /**
             * @brief Records the unconjugated rank-1 update A = alpha * x * y^T + A.
             * @throws std::invalid_argument on an operand or a scalar the call does not accept.
             */
            TaskHandle geru(const Tensor& x, const Tensor& y, Tensor& A, BlasScalar alpha = 1.0);

            /**
             * @brief Records the conjugated rank-1 update A = alpha * x * y^H + A.
             * @throws std::invalid_argument on an operand or a scalar the call does not accept.
             */
            TaskHandle gerc(const Tensor& x, const Tensor& y, Tensor& A, BlasScalar alpha = 1.0);

            /**
             * @brief Records y = alpha * A * x + beta * y for a symmetric A.
             * @param uplo The triangle of A that is read.
             * @param beta Exactly zero skips reading y.
             * @throws std::invalid_argument on an operand, flag or scalar the call does not accept.
             */
            TaskHandle symv(const Tensor& A, const Tensor& x, Tensor& y, Core::Uplo uplo,
                            BlasScalar alpha = 1.0, BlasScalar beta = 0.0);

            /**
             * @brief Records y = alpha * A * x + beta * y for a Hermitian A.
             * @param uplo The triangle of A that is read; its diagonal's imaginary part is not.
             * @param beta Exactly zero skips reading y.
             * @throws std::invalid_argument on an operand, flag or scalar the call does not accept.
             */
            TaskHandle hemv(const Tensor& A, const Tensor& x, Tensor& y, Core::Uplo uplo,
                            BlasScalar alpha = 1.0, BlasScalar beta = 0.0);

            /**
             * @brief Records x = op(A) * x for a triangular A.
             * @throws std::invalid_argument on an operand or a flag the call does not accept.
             */
            TaskHandle trmv(const Tensor& A, Tensor& x, Core::Uplo uplo,
                            Core::Transpose trans = Core::Transpose::NO_TRANS,
                            Core::Diag diag = Core::Diag::NON_UNIT);

            /**
             * @brief Records the solution of op(A) * x = b for a triangular A, overwriting b.
             * @param b Contiguous.
             * @throws std::invalid_argument on an operand or a flag the call does not accept.
             */
            TaskHandle trsv(const Tensor& A, Tensor& b, Core::Uplo uplo,
                            Core::Transpose trans = Core::Transpose::NO_TRANS,
                            Core::Diag diag = Core::Diag::NON_UNIT);

            /**
             * @brief Records the symmetric rank-1 update A = alpha * x * x^T + A.
             * @param uplo The triangle of A that is updated.
             * @throws std::invalid_argument on an operand, flag or scalar the call does not accept.
             */
            TaskHandle syr(const Tensor& x, Tensor& A, Core::Uplo uplo, BlasScalar alpha = 1.0);

            /**
             * @brief Records the Hermitian rank-1 update A = alpha * x * x^H + A.
             * @param uplo The triangle of A that is updated; its diagonal is left real.
             * @throws std::invalid_argument on an operand or a flag the call does not accept.
             */
            TaskHandle her(const Tensor& x, Tensor& A, Core::Uplo uplo, double alpha = 1.0);

            /**
             * @brief Records the symmetric rank-2 update A = alpha * x * y^T + alpha * y * x^T + A.
             * @param uplo The triangle of A that is updated.
             * @throws std::invalid_argument on an operand, flag or scalar the call does not accept.
             */
            TaskHandle syr2(const Tensor& x, const Tensor& y, Tensor& A, Core::Uplo uplo,
                            BlasScalar alpha = 1.0);

            /**
             * @brief Records the Hermitian rank-2 update
             *        A = alpha * x * y^H + conj(alpha) * y * x^H + A.
             * @param uplo The triangle of A that is updated; its diagonal is left real.
             * @throws std::invalid_argument on an operand, flag or scalar the call does not accept.
             */
            TaskHandle her2(const Tensor& x, const Tensor& y, Tensor& A, Core::Uplo uplo,
                            BlasScalar alpha = 1.0);

        protected:
            /** @brief Binds the recorder every operation records on; it must outlive this view. */
            explicit Level2(TaskRecorder& recorder) noexcept
                : recorder_(recorder)
            {

            }

            /** @brief Copies the binding. */
            Level2(const Level2&) noexcept = default;

            /** @brief Moves the binding. */
            Level2(Level2&&) noexcept = default;

            /** @brief Refused: a view cannot be re-bound. */
            Level2& operator=(const Level2&) = delete;

            /** @brief Refused: a view cannot be re-bound. */
            Level2& operator=(Level2&&) = delete;

            /** @brief Destroys the view; the recorder is not owned. */
            ~Level2() = default;

        private:
            /// @brief The recorder every operation of this level records on.
            TaskRecorder& recorder_;
    };
} // namespace SushiBLAS
