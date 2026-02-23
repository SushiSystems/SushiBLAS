/**************************************************************************/
/* linalg.hpp                                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                   	  */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

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
