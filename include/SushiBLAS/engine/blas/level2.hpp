/**************************************************************************/
/* level2.hpp                                                             */
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
     * @class Level2
     * @brief Matrix-Vector operations (BLAS Level 2).
     * 
     * This class provides an interface for standard BLAS Level 2 operations, 
     * which involve matrix-vector computations. All operations are executed 
     * asynchronously using the SushiBLAS task graph system.
     */
    class Level2 
    {
        protected:
            explicit Level2(Engine& e) : engine_(e) {}
            Engine& engine_;

        public:
            /** 
             * @brief General Matrix-Vector Multiplication (GEMV).
             * 
             * Computes the operation: y = alpha * op(A) * x + beta * y.
             * Performs a general matrix-vector multiplication with scaling factors.
             * 
             * @param A Input matrix A.
             * @param x Input vector x.
             * @param y Input/Output vector y.
             * @param transA Whether to transpose A (op(A) = A^T if true, else A).
             * @param alpha Scalar multiplier for A*x.
             * @param beta Scalar multiplier for y.
             * @return sycl::event representing the completion of the operation.
             */
            sycl::event gemv(const Tensor& A, const Tensor& x, Tensor& y,
                             bool transA = false, float alpha = 1.0f, float beta = 0.0f);

            /** 
             * @brief General Rank-1 Update (GER).
             * 
             * Computes the operation: A = alpha * x * y^H + A.
             * Performs a rank-1 update of general matrix A using vectors x and y.
             * 
             * @param x Input vector x.
             * @param y Input vector y.
             * @param A Input/Output matrix A.
             * @param alpha Scalar multiplier.
             * @return sycl::event representing the completion of the operation.
             */
            sycl::event ger(const Tensor& x, const Tensor& y, Tensor& A, float alpha = 1.0f);

            /**
             * @brief Symmetric Matrix-Vector Multiplication (SYMV).
             * 
             * Computes the operation: y = alpha * A * x + beta * y where A is symmetric.
             * Efficiently performs matrix-vector multiplication for symmetric matrices.
             * 
             * @param A Input symmetric matrix A.
             * @param x Input vector x.
             * @param y Input/Output vector y.
             * @param upper Whether to use the upper triangle of A (true) or lower (false).
             * @param alpha Scalar multiplier.
             * @param beta Scalar multiplier.
             * @return sycl::event representing the completion of the operation.
             */
            sycl::event symv(const Tensor& A, const Tensor& x, Tensor& y, 
                             bool upper = false, float alpha = 1.0f, float beta = 0.0f);

            /**
             * @brief Triangular Matrix-Vector Multiplication (TRMV).
             * 
             * Computes the operation: x = op(A) * x where A is triangular.
             * Performs a matrix-vector multiplication where A is a triangular matrix.
             * 
             * @param A Input triangular matrix A.
             * @param x Input/Output vector x.
             * @param upper Whether A is upper triangular (true) or lower (false).
             * @param transA Whether to transpose A.
             * @param unit_diag Whether A has unit diagonal (all ones on diagonal).
             * @return sycl::event representing the completion of the operation.
             */
            sycl::event trmv(const Tensor& A, Tensor& x, 
                             bool upper = false, bool transA = false, bool unit_diag = false);

            /**
             * @brief Triangular Solve (TRSV).
             * 
             * Solves the linear system: op(A) * x = b for x, where A is triangular.
             * The result x overwrites the input vector b.
             * 
             * @param A Input triangular matrix A.
             * @param b Input/Output vector (initially b, overwritten by solution x).
             * @param upper Whether A is upper triangular (true) or lower (false).
             * @param transA Whether to transpose A.
             * @param unit_diag Whether A has unit diagonal.
             * @return sycl::event representing the completion of the operation.
             */
            sycl::event trsv(const Tensor& A, Tensor& b, 
                             bool upper = false, bool transA = false, bool unit_diag = false);

            /**
             * @brief Symmetric Rank-1 Update (SYR).
             * 
             * Computes the operation: A = alpha * x * x^H + A where A is symmetric.
             * Performs a symmetric rank-1 update of matrix A.
             * 
             * @param x Input vector x.
             * @param A Input/Output symmetric matrix A.
             * @param upper Whether to update the upper triangle (true) or lower (false).
             * @param alpha Scalar multiplier.
             * @return sycl::event representing the completion of the operation.
             */
            sycl::event syr(const Tensor& x, Tensor& A, bool upper = false, float alpha = 1.0f);

            /**
             * @brief Symmetric Rank-2 Update (SYR2).
             * 
             * Computes the operation: A = alpha * x * y^H + alpha * y * x^H + A where A is symmetric.
             * Performs a symmetric rank-2 update of matrix A.
             * 
             * @param x Input vector x.
             * @param y Input vector y.
             * @param A Input/Output symmetric matrix A.
             * @param upper Whether to update the upper triangle (true) or lower (false).
             * @param alpha Scalar multiplier.
             * @return sycl::event representing the completion of the operation.
             */
            sycl::event syr2(const Tensor& x, const Tensor& y, Tensor& A, bool upper = false, float alpha = 1.0f);
    };
}
