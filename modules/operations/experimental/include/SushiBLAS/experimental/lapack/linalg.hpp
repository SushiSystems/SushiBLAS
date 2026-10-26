/**************************************************************************/
/* linalg.hpp                                                             */
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
 * @file linalg.hpp
 * @brief Declares LinalgOps, the dense linear algebra operations: batched LU and GMRES.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/operation_options.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>

namespace SushiBLAS
{
    /**
     * @brief Holds the node options and the stopping rule of one gmres solve.
     * @see modules/operations/experimental/README.md
     */
    struct GmresOptions
    {
            OperationOptions node{"linalg.gmres"}; /**< Name and class of the node. */
            std::size_t restart = 30;              /**< Krylov vectors kept before a restart. */
            std::size_t max_iterations = 300;      /**< Inner iterations across all restarts. */
            double relative_tolerance = 1e-6;      /**< Stop once |r| / |b| falls below it. */
            std::size_t block_size = 1;            /**< Diagonal block of block-Jacobi. */
    };

    /**
     * @class LinalgOps
     * @brief Records dense linear algebra operations; only the LU and GMRES ones are defined.
     * @see modules/operations/experimental/README.md
     */
    class LinalgOps final
    {
        public:
            /** @brief Builds a view that records through recorder, which must outlive the view. */
            explicit LinalgOps(TaskRecorder& recorder) noexcept : recorder_(recorder) {}

            LinalgOps(const LinalgOps&) = default;
            LinalgOps& operator=(const LinalgOps&) = delete;
            LinalgOps(LinalgOps&&) = default;
            LinalgOps& operator=(LinalgOps&&) = delete;
            ~LinalgOps() = default;

            /**
             * @brief Matrix Inversion.
             * Computes A^-1. Overwrites input A with its inverse.
             * @param A Input square matrix.
             * @return The handle of the recorded task.
             */
            TaskHandle inv(Tensor& A);

            /**
             * @brief Records P A = L U with partial pivoting for each matrix of a batch, in place.
             * @param matrices FLOAT32 or FLOAT64, batch row-major n * n matrices, replaced by L U.
             * @param pivots UINT32, n per matrix; element i is the row swapped with row i at step i
             * @param singular UINT32, one per matrix; 1 when a pivot is not above n * eps * max|a|.
             * @throws std::invalid_argument at record time on a mismatched or non-dense operand.
             * @see modules/operations/experimental/README.md
             */
            TaskHandle lu_factor(Tensor& matrices, Tensor& pivots, Tensor& singular,
                                 std::size_t n, std::size_t batch);

            /**
             * @brief Records lu_factor under the name and class options states.
             * @throws std::invalid_argument as the form without options, and for an empty name.
             */
            TaskHandle lu_factor(Tensor& matrices, Tensor& pivots, Tensor& singular,
                                 std::size_t n, std::size_t batch, const OperationOptions& options);

            /**
             * @brief Records the solve of A X = B for each matrix lu_factor factored, in place.
             * @param rhs The dtype of factors, batch row-major n * rhs_count blocks; replaced by X.
             * @pre No matrix of the batch was reported singular.
             * @throws std::invalid_argument at record time on a mismatched or non-dense operand.
             * @see modules/operations/experimental/README.md
             */
            TaskHandle lu_solve(const Tensor& factors, const Tensor& pivots, Tensor& rhs,
                                std::size_t n, std::size_t rhs_count, std::size_t batch);

            /**
             * @brief Records lu_solve under the name and class options states.
             * @throws std::invalid_argument as the form without options, and for an empty name.
             */
            TaskHandle lu_solve(const Tensor& factors, const Tensor& pivots, Tensor& rhs,
                                std::size_t n, std::size_t rhs_count, std::size_t batch,
                                const OperationOptions& options);

            /**
             * @brief Records a restarted, block-Jacobi preconditioned GMRES solve of A x = b.
             * @param matrix FLOAT32 or FLOAT64, row-major n * n, where n is rhs's element count.
             * @param solution The starting guess on entry, the result on exit.
             * @param iterations UINT32, the inner iterations taken; residual gets |b - A x| / |b|.
             * @throws std::invalid_argument on a mismatched operand or option.
             * @see modules/operations/experimental/README.md
             */
            TaskHandle gmres(const Tensor& matrix, const Tensor& rhs, Tensor& solution,
                             Tensor& iterations, Tensor& residual, const GmresOptions& options);

            /**
             * @brief QR Decomposition (A = Q*R).
             * Factorizes matrix A into an orthogonal matrix Q and an upper triangular matrix R.
             * @param A Input matrix.
             * @param Q Output orthogonal matrix.
             * @param R Output upper triangular matrix.
             * @return The handle of the recorded task.
             */
            TaskHandle qr(const Tensor& A, Tensor& Q, Tensor& R);

            /**
             * @brief Factorizes A into singular values and singular vectors: A = U*S*V^H.
             */
            TaskHandle svd(const Tensor& A, Tensor& U, Tensor& S, Tensor& V);

            /**
             * @brief Eigendecomposition for Symmetric Matrices.
             * Computes eigenvalues (W) and optionally eigenvectors (V) of a symmetric matrix A.
             * @param A Input symmetric matrix.
             * @param W Output vector of eigenvalues.
             * @param V Optional output matrix of eigenvectors.
             * @return The handle of the recorded task.
             */
            TaskHandle eigh(const Tensor& A, Tensor& W, Tensor* V = nullptr);

            /**
             * @brief General Eigenvalue Problem.
             * Computes eigenvalues for a general non-symmetric square matrix A.
             * @param A Input square matrix.
             * @param W Output vector of eigenvalues (complex values).
             * @return The handle of the recorded task.
             */
            TaskHandle eig(const Tensor& A, Tensor& W);

            /**
             * @brief Numerical Determinant.
             * Computes the determinant of matrix A.
             * @param A Input square matrix.
             * @param result Scalar output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle det(const Tensor& A, Tensor& result);

            /**
             * @brief Solve Linear System (Ax = B).
             * Solves a system of linear equations for x.
             * @param A Coefficient matrix.
             * @param B Right-hand side matrix (overwritten by solution x).
             * @return The handle of the recorded task.
             */
            TaskHandle solve(const Tensor& A, Tensor& B);

        private:
            /// @brief The recorder every operation of this view records through.
            TaskRecorder& recorder_;
    };

} // namespace SushiBLAS
