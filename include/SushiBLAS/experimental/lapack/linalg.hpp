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
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

// EXPERIMENTAL - quarantined per SB-2/D1. Not included from engine.hpp and not
// reachable via Engine::linalg() - that accessor was removed. No .cpp
// implementation exists for any method declared here. The LAPACK backend
// strategy is a named follow-up (P8), not solved by this file's presence.

#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor.hpp>

namespace SushiBLAS
{
    class Engine;

    /**
     * @class LinalgOps
     * @brief High-level linear algebra operations (LAPACK).
     *
     * Provides matrix solvers, decompositions, and structural matrix analysis.
     */
    class LinalgOps
    {
        public:
            /**
             * @brief Construct LinalgOps with a reference to the engine.
             * @param e The SushiBLAS engine.
             */
            explicit LinalgOps(Engine& e) : engine_(e) {}

            /**
             * @brief Matrix Inversion.
             * Computes A^-1. Overwrites input A with its inverse.
             * @param A Input square matrix.
             * @return sycl::event representing task completion.
             */
            sycl::event inv(Tensor& A);

            /**
             * @brief LU Decomposition (A = P*L*U).
             * Factorizes matrix A into a permutation matrix P, a unit lower
             * triangular matrix L, and an upper triangular matrix U.
             * @param A Input matrix (overwritten by L and U).
             * @param ipiv Output pivot indices.
             * @return sycl::event.
             */
            sycl::event lu(Tensor& A, Tensor& ipiv);

            /**
             * @brief QR Decomposition (A = Q*R).
             * Factorizes matrix A into an orthogonal matrix Q and an upper triangular matrix R.
             * @param A Input matrix.
             * @param Q Output orthogonal matrix.
             * @param R Output upper triangular matrix.
             * @return sycl::event.
             */
            sycl::event qr(const Tensor& A, Tensor& Q, Tensor& R);

            /**
             * @brief Singular Value Decomposition (A = U*S*V^H).
             * Factorizes matrix A into singular values and singular vectors.
             * @param A Input matrix.
             * @param U Output unitary matrix.
             * @param S Output vector of singular values.
             * @param V Output unitary matrix.
             * @return sycl::event.
             */
            sycl::event svd(const Tensor& A, Tensor& U, Tensor& S, Tensor& V);

            /**
             * @brief Eigendecomposition for Symmetric Matrices.
             * Computes eigenvalues (W) and optionally eigenvectors (V) of a symmetric matrix A.
             * @param A Input symmetric matrix.
             * @param W Output vector of eigenvalues.
             * @param V Optional output matrix of eigenvectors.
             * @return sycl::event.
             */
            sycl::event eigh(const Tensor& A, Tensor& W, Tensor* V = nullptr);

            /**
             * @brief General Eigenvalue Problem.
             * Computes eigenvalues for a general non-symmetric square matrix A.
             * @param A Input square matrix.
             * @param W Output vector of eigenvalues (complex values).
             * @return sycl::event.
             */
            sycl::event eig(const Tensor& A, Tensor& W);

            /**
             * @brief Numerical Determinant.
             * Computes the determinant of matrix A.
             * @param A Input square matrix.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event det(const Tensor& A, Tensor& result);

            /**
             * @brief Solve Linear System (Ax = B).
             * Solves a system of linear equations for x.
             * @param A Coefficient matrix.
             * @param B Right-hand side matrix (overwritten by solution x).
             * @return sycl::event.
             */
            sycl::event solve(const Tensor& A, Tensor& B);

        private:
            Engine& engine_;
    };

} // namespace SushiBLAS
