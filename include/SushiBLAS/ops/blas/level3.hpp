/**************************************************************************/
/* level3.hpp                                                             */
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
#include <sycl/sycl.hpp>

namespace SushiBLAS 
{
    class Engine;

    /**
     * @brief Matrix-Matrix operations.
     */
    class Level3 
    {
        protected:
            explicit Level3(Engine& e) : engine_(e) {}
            Engine& engine_;

        public:
            /**
             * @brief General Matrix-Matrix Multiplication
             * Computes C = alpha * op(A) * op(B) + beta * C
             * 
             * @param A Input matrix A
             * @param B Input matrix B
             * @param C Output matrix C
             * @param transA Whether to transpose A
             * @param transB Whether to transpose B
             * @param alpha Scalar multiplier for A*B
             * @param beta Scalar multiplier for C
             */
            sycl::event gemm(const Tensor& A, const Tensor& B, Tensor& C, 
                            bool transA = false, bool transB = false,
                            float alpha = 1.0f, float beta = 0.0f);

            /**
             * @brief Triangular Solve with Multiple Right-Hand Sides
             * Solves op(A)*X = alpha*B or X*op(A) = alpha*B and overwrites B with X.
             * 
             * @param A Input triangular matrix A
             * @param B Input/Output matrix B (B is overwritten with solution X)
             * @param left_side Whether A is on the left (op(A)*X) or right (X*op(A))
             * @param upper Whether A is upper triangular
             * @param transA Whether to transpose A
             * @param unit_diag Whether A has unit diagonal (no scaling)
             * @param alpha Scalar multiplier
             */
            sycl::event trsm(const Tensor& A, Tensor& B, 
                      bool left_side = true, bool upper = false, 
                      bool transA = false, bool unit_diag = false, 
                      float alpha = 1.0f);

            /**
             * @brief Symmetric Rank-k Update
             * Computes C = alpha * op(A) * op(A)^T + beta * C
             * 
             * @param A Input matrix A
             * @param C Input/Output matrix C (C is updated in-place)
             * @param upper Whether to store the upper triangle of C
             * @param transA Whether to transpose A
             * @param alpha Scalar multiplier for A*A^T
             * @param beta Scalar multiplier for C
             */
            sycl::event syrk(const Tensor& A, Tensor& C, 
                      bool upper = false, bool transA = false, 
                      float alpha = 1.0f, float beta = 0.0f);
    };
}
