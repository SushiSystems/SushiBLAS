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

#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor.hpp>

namespace SushiBLAS 
{
    class Engine;

    /**
     * @class Level3
     * @brief Matrix-Matrix operations (BLAS Level 3).
     * 
     * This class provides an interface for standard BLAS Level 3 operations, 
     * which involve matrix-matrix computations. These operations are typically 
     * the most computationally intensive and benefit greatly from optimized 
     * implementations like oneMKL. All operations are executed asynchronously 
     * using the SushiBLAS task graph system.
     */
    class Level3 
    {
        protected:
            explicit Level3(Engine& e) : engine_(e) {}
            Engine& engine_;

        public:
            /**
             * @brief General Matrix-Matrix Multiplication (GEMM).
             * 
             * Computes the operation: C = alpha * op(A) * op(B) + beta * C.
             * This supports batching automatically if A, B, and C have rank > 2.
             * 
             * @param A Input matrix A.
             * @param B Input matrix B.
             * @param C Output matrix C.
             * @param transA Whether to transpose A (op(A) = A^T if true, else A).
             * @param transB Whether to transpose B (op(B) = B^T if true, else B).
             * @param alpha Scalar multiplier for A*B.
             * @param beta Scalar multiplier for C.
             * @return sycl::event representing the completion of the operation.
             */
            sycl::event gemm(const Tensor& A, const Tensor& B, Tensor& C, 
                            bool transA = false, bool transB = false,
                            double alpha = 1.0, double beta = 0.0);

            /**
             * @brief Triangular Solve with Multiple Right-Hand Sides (TRSM).
             * 
             * Solves one of the following systems:
             * op(A)*X = alpha*B   (if left_side=true)
             * X*op(A) = alpha*B   (if left_side=false)
             * where A is a triangular matrix. The result X overwrites matrix B.
             * 
             * @param A Input triangular matrix A.
             * @param B Input/Output matrix B (initially right-hand sides, overwritten by solution X).
             * @param left_side Whether A is on the left (true) or right (false) of X.
             * @param upper Whether A is upper triangular (true) or lower (false).
             * @param transA Whether to transpose A.
             * @param unit_diag Whether A has unit diagonal.
             * @param alpha Scalar multiplier for B.
             * @return sycl::event representing the completion of the operation.
             */
            sycl::event trsm(const Tensor& A, Tensor& B, 
                      bool left_side = true, bool upper = false, 
                      bool transA = false, bool unit_diag = false, 
                      double alpha = 1.0);

            /**
             * @brief Symmetric Rank-k Update (SYRK).
             * 
             * Computes one of the following updates:
             * C = alpha * A * A^T + beta * C   (if transA=false)
             * C = alpha * A^T * A + beta * C   (if transA=true)
             * where C is a symmetric matrix.
             * 
             * @param A Input matrix A.
             * @param C Input/Output symmetric matrix C (only half is updated/stored).
             * @param upper Whether to store the result in the upper triangle of C (true) or lower (false).
             * @param transA Whether to transpose A in the update.
             * @param alpha Scalar multiplier for the update term.
             * @param beta Scalar multiplier for C.
             * @return sycl::event representing the completion of the operation.
             */
            sycl::event syrk(const Tensor& A, Tensor& C, 
                      bool upper = false, bool transA = false, 
                      double alpha = 1.0, double beta = 0.0);
    };
}
