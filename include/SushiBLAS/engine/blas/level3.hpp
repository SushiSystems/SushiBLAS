/****************************************************************/
/* level3.hpp                                                   */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file level3.hpp
 * @brief Declares Level3, the matrix-matrix BLAS operations.
 * @author Mustafa Garip
 */

#pragma once

#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor.hpp>

namespace SushiBLAS 
{
    class Engine;

    /**
     * @brief Records the BLAS Level 3 matrix-matrix operations on the engine's task graph.
     * @see include/SushiBLAS/engine/blas/README.md
     */
    class Level3 
    {
        protected:
            explicit Level3(Engine& e) : engine_(e) {}
            Engine& engine_;

        public:
            /**
             * @brief Computes C = alpha * op(A) * op(B) + beta * C, batched when the operands have rank > 2.
             */
            sycl::event gemm(const Tensor& A, const Tensor& B, Tensor& C, 
                            bool transA = false, bool transB = false,
                            double alpha = 1.0, double beta = 0.0);

            /**
             * @brief Solves a triangular system with multiple right-hand sides, overwriting B with X.
             * @param left_side Solves op(A) * X = alpha * B when true, X * op(A) = alpha * B when false.
             * @param upper Treats A as upper triangular when true, lower when false.
             */
            sycl::event trsm(const Tensor& A, Tensor& B, 
                      bool left_side = true, bool upper = false, 
                      bool transA = false, bool unit_diag = false, 
                      double alpha = 1.0);

            /**
             * @brief Computes the symmetric rank-k update of C from A.
             * @param upper Stores the result in the upper triangle of C when true, the lower when false.
             * @param transA Computes alpha * A^T * A + beta * C when true, alpha * A * A^T + beta * C when false.
             */
            sycl::event syrk(const Tensor& A, Tensor& C, 
                      bool upper = false, bool transA = false, 
                      double alpha = 1.0, double beta = 0.0);
    };
}
