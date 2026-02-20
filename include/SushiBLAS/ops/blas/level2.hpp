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

#include <SushiBLAS/tensor.hpp>

namespace SushiBLAS 
{
    class Engine;

    /**
     * @brief Matrix-Vector operations.
     */
    class Level2 
    {
        protected:
            explicit Level2(Engine& e) : engine_(e) {}
            Engine& engine_;

        public:
            /** 
             * @brief y = alpha*A*x + beta*y (General Matrix-Vector Multiplication)
             * Used extensively in Neural Networks (Linear Layers) and Physics (Force/Momentum calculations).
             * 
             * @param alpha Scalar multiplier for A*x
             * @param A Input matrix
             * @param x Input vector
             * @param beta Scalar multiplier for y
             * @param y Input/Output vector
             * @param transA Whether to transpose A
             */
            void gemv(const Tensor& A, const Tensor& x, Tensor& y,
                      bool transA = false, float alpha = 1.0f, float beta = 0.0f);

            /** 
             * @brief A = alpha*x*y^H + A (General Rank-1 Update)
             * Fundamental in updating covariance matrices and outer product operations.
             * 
             * @param x Input vector
             * @param y Input vector
             * @param A Input/Output matrix
             * @param alpha Scalar multiplier
             */
            void ger(const Tensor& x, const Tensor& y, Tensor& A, float alpha = 1.0f);

            /**
             * @brief y = alpha*A*x + beta*y (Symmetric Matrix-Vector Multiplication)
             * Crucial for Physics simulations involving symmetric tensors (stress, inertia).
             * 
             * @param A Symmetric input matrix
             * @param x Input vector
             * @param y Input/Output vector
             * @param upper Use upper or lower triangle of A
             * @param alpha Scalar multiplier
             * @param beta Scalar multiplier
             */
            void symv(const Tensor& A, const Tensor& x, Tensor& y, 
                      bool upper = false, float alpha = 1.0f, float beta = 0.0f);

            /**
             * @brief x = op(A)*x (Triangular Matrix-Vector Multiplication)
             * 
             * @param A Triangular matrix
             * @param x Input/Output vector
             * @param upper Use upper or lower triangle
             * @param transA Transpose A
             * @param unit_diag Whether A has unit diagonal
             */
            void trmv(const Tensor& A, Tensor& x, 
                      bool upper = false, bool transA = false, bool unit_diag = false);

            /**
             * @brief Solves op(A)*x = b (Triangular Solve)
             * 
             * @param A Triangular matrix
             * @param b Input/Output vector (result x overwrites b)
             * @param upper Use upper or lower triangle
             * @param transA Transpose A
             * @param unit_diag Whether A has unit diagonal
             */
            void trsv(const Tensor& A, Tensor& b, 
                      bool upper = false, bool transA = false, bool unit_diag = false);

            /**
             * @brief A = alpha*x*x^H + A (Symmetric Rank-1 Update)
             * 
             * @param x Input vector
             * @param A Input/Output symmetric matrix
             * @param upper Update upper or lower triangle
             * @param alpha Scalar multiplier
             */
            void syr(const Tensor& x, Tensor& A, bool upper = false, float alpha = 1.0f);

            /**
             * @brief A = alpha*x*y^H + alpha*y*x^H + A (Symmetric Rank-2 Update)
             * 
             * @param x Input vector
             * @param y Input vector
             * @param A Input/Output symmetric matrix
             * @param upper Update upper or lower triangle
             * @param alpha Scalar multiplier
             */
            void syr2(const Tensor& x, const Tensor& y, Tensor& A, bool upper = false, float alpha = 1.0f);
    };
}
