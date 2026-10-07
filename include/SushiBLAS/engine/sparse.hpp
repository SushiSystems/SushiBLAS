/**************************************************************************/
/* sparse.hpp                                                             */
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
 * @file sparse.hpp
 * @brief Declares SparseOps, the view that records the sparse BLAS operations.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/sparse_matrix.hpp>
#include <SushiBLAS/sparse_vector.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine/blas/scalar.hpp>

namespace SushiBLAS
{
    /**
     * @brief Records the sparse-dense level 1 routines and the products and solves of a CSR
     *        matrix through a task recorder.
     * @see include/SushiBLAS/engine/sparse/README.md
     */
    class SparseOps final
    {
        public:
            /** @brief Builds a view that records through recorder, which must outlive the view. */
            explicit SparseOps(TaskRecorder& recorder) noexcept
                : recorder_(recorder)
            {

            }

            /** @brief Builds a view that records on engine's own graph. */
            explicit SparseOps(Engine& engine) noexcept
                : SparseOps(engine.get_recorder())
            {

            }

            /** @brief Copies the binding. */
            SparseOps(const SparseOps&) noexcept = default;

            /** @brief Moves the binding. */
            SparseOps(SparseOps&&) noexcept = default;

            /** @brief Refused: a view cannot be re-bound. */
            SparseOps& operator=(const SparseOps&) = delete;

            /** @brief Refused: a view cannot be re-bound. */
            SparseOps& operator=(SparseOps&&) = delete;

            /** @brief Destroys the view; the recorder is not owned. */
            ~SparseOps() = default;

            /**
             * @brief Records y[p] = alpha * x[p] + y[p] over the stored positions p of x.
             * @param y Dense, of the length x.size.
             * @throws std::invalid_argument on an operand or a scalar the call does not accept.
             */
            TaskHandle axpyi(BlasScalar alpha, const SparseVector& x, Tensor& y);

            /**
             * @brief Records the dot product of a real sparse x and a dense y into result.
             * @param result Scalar tensor of the operands' dtype.
             * @throws std::invalid_argument on a complex operand, or one the call does not accept.
             */
            TaskHandle doti(const SparseVector& x, const Tensor& y, Tensor& result);

            /**
             * @brief Records the unconjugated dot product x^T * y of a sparse x and a dense y.
             * @param result Scalar tensor of the operands' dtype.
             * @throws std::invalid_argument on an operand the call does not accept.
             */
            TaskHandle dotui(const SparseVector& x, const Tensor& y, Tensor& result);

            /**
             * @brief Records the conjugated dot product x^H * y of a sparse x and a dense y.
             * @param result Scalar tensor of the operands' dtype.
             * @throws std::invalid_argument on an operand the call does not accept.
             */
            TaskHandle dotci(const SparseVector& x, const Tensor& y, Tensor& result);

            /**
             * @brief Records a gather: the values of x become the elements of y at x's positions.
             * @param x Its indices are read and its values written.
             * @throws std::invalid_argument on an operand the call does not accept.
             */
            TaskHandle gthr(const Tensor& y, SparseVector& x);

            /**
             * @brief Records a gather that also sets the gathered elements of y to zero.
             * @param x Its indices are read and its values written.
             * @throws std::invalid_argument on an operand the call does not accept.
             */
            TaskHandle gthrz(Tensor& y, SparseVector& x);

            /**
             * @brief Records a scatter: the elements of y at x's positions become the values of x.
             * @param y Dense, of the length x.size; its other elements are left as they are.
             * @throws std::invalid_argument on an operand the call does not accept.
             */
            TaskHandle sctr(const SparseVector& x, Tensor& y);

            /**
             * @brief Records the plane rotation x[p] = c * x[p] + s * y[p],
             *        y[p] = c * y[p] - s * x[p] over the stored positions p of x.
             * @param c Real for every dtype, as is s.
             * @throws std::invalid_argument on an operand the call does not accept.
             */
            TaskHandle roti(SparseVector& x, Tensor& y, double c, double s);

            /**
             * @brief Records y = alpha * op(A) * x + beta * y for a CSR matrix A.
             * @param y Must not overlap x or an array of A.
             * @param beta Exactly zero skips reading y.
             * @throws std::invalid_argument on an operand, flag or scalar the call does not accept.
             */
            TaskHandle spmv(const SparseMatrix& A, const Tensor& x, Tensor& y,
                            Core::Transpose trans = Core::Transpose::NO_TRANS,
                            BlasScalar alpha = 1.0, BlasScalar beta = 0.0);

            /**
             * @brief Records y = alpha * A * x + beta * y for a symmetric A stored as one triangle.
             * @param A Square; the entries outside the triangle uplo names are not used.
             * @param beta Exactly zero skips reading y.
             * @throws std::invalid_argument on an operand, flag or scalar the call does not accept.
             */
            TaskHandle spsymv(const SparseMatrix& A, const Tensor& x, Tensor& y, Core::Uplo uplo,
                              BlasScalar alpha = 1.0, BlasScalar beta = 0.0);

            /**
             * @brief Records y = alpha * A * x + beta * y for a Hermitian A stored as one triangle.
             * @param A Square; the other triangle and the diagonal's imaginary part are not used.
             * @param beta Exactly zero skips reading y.
             * @throws std::invalid_argument on an operand, flag or scalar the call does not accept.
             */
            TaskHandle sphemv(const SparseMatrix& A, const Tensor& x, Tensor& y, Core::Uplo uplo,
                              BlasScalar alpha = 1.0, BlasScalar beta = 0.0);

            /**
             * @brief Records C = alpha * op(A) * B + beta * C for a CSR matrix A and dense B, C.
             * @param C Rank 2; must not overlap B or an array of A.
             * @param beta Exactly zero skips reading C.
             * @throws std::invalid_argument on an operand, flag or scalar the call does not accept.
             */
            TaskHandle spmm(const SparseMatrix& A, const Tensor& B, Tensor& C,
                            Core::Transpose trans = Core::Transpose::NO_TRANS,
                            BlasScalar alpha = 1.0, BlasScalar beta = 0.0);

            /**
             * @brief Records the solution of op(A) * x = b for a triangular CSR A, overwriting b.
             * @param A Square; the entries outside the triangle uplo names are not used.
             * @param b Contiguous.
             * @throws std::invalid_argument on an operand or a flag the call does not accept.
             */
            TaskHandle sptrsv(const SparseMatrix& A, Tensor& b, Core::Uplo uplo,
                              Core::Transpose trans = Core::Transpose::NO_TRANS,
                              Core::Diag diag = Core::Diag::NON_UNIT);

        private:
            /// @brief The recorder every operation of this view records through.
            TaskRecorder& recorder_;
    };
} // namespace SushiBLAS
