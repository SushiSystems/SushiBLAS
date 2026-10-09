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
 * @brief Declares LinalgOps, the experimental LAPACK operations that have no implementation yet.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>

namespace SushiBLAS
{
    /**
     * @class LinalgOps
     * @brief High-level linear algebra operations (LAPACK).
     *
     * Provides matrix solvers, decompositions, and structural matrix analysis.
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
             * @brief LU Decomposition (A = P*L*U).
             * Factorizes matrix A into a permutation matrix P, a unit lower
             * triangular matrix L, and an upper triangular matrix U.
             * @param A Input matrix (overwritten by L and U).
             * @param ipiv Output pivot indices.
             * @return The handle of the recorded task.
             */
            TaskHandle lu(Tensor& A, Tensor& ipiv);

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
