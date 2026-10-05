/****************************************************************/
/* level2.hpp                                                   */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file level2.hpp
 * @brief Declares Level2, the matrix-vector BLAS operations.
 * @author Mustafa Garip
 */

#pragma once

#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor.hpp>

namespace SushiBLAS 
{
    class Engine;

    /**
     * @brief Records the BLAS Level 2 matrix-vector operations on the engine's task graph.
     * @see include/SushiBLAS/engine/blas/README.md
     */
    class Level2 
    {
        protected:
            explicit Level2(Engine& e) : engine_(e) {}
            Engine& engine_;

        public:
            /**
             * @brief Computes y = alpha * op(A) * x + beta * y.
             * @param transA Uses A^T as op(A) when true.
             */
            sycl::event gemv(const Tensor& A, const Tensor& x, Tensor& y,
                             bool transA = false, double alpha = 1.0, double beta = 0.0);

            /**
             * @brief Computes the rank-1 update A = alpha * x * y^H + A.
             */
            sycl::event ger(const Tensor& x, const Tensor& y, Tensor& A, double alpha = 1.0);

            /**
             * @brief Computes y = alpha * A * x + beta * y for a symmetric A.
             * @param upper Reads the upper triangle of A when true, the lower when false.
             */
            sycl::event symv(const Tensor& A, const Tensor& x, Tensor& y, 
                             bool upper = false, double alpha = 1.0, double beta = 0.0);

            /**
             * @brief Computes x = op(A) * x for a triangular A.
             * @param upper Treats A as upper triangular when true, lower when false.
             * @param unit_diag Treats the diagonal of A as all ones when true.
             */
            sycl::event trmv(const Tensor& A, Tensor& x, 
                             bool upper = false, bool transA = false, bool unit_diag = false);

            /**
             * @brief Solves op(A) * x = b for a triangular A, overwriting b with x.
             * @param upper Treats A as upper triangular when true, lower when false.
             * @param unit_diag Treats the diagonal of A as all ones when true.
             */
            sycl::event trsv(const Tensor& A, Tensor& b, 
                             bool upper = false, bool transA = false, bool unit_diag = false);

            /**
             * @brief Computes the symmetric rank-1 update A = alpha * x * x^H + A.
             * @param upper Updates the upper triangle when true, the lower when false.
             */
            sycl::event syr(const Tensor& x, Tensor& A, bool upper = false, double alpha = 1.0);

            /**
             * @brief Computes the symmetric rank-2 update A = alpha * x * y^H + alpha * y * x^H + A.
             * @param upper Updates the upper triangle when true, the lower when false.
             */
            sycl::event syr2(const Tensor& x, const Tensor& y, Tensor& A, bool upper = false, double alpha = 1.0);
    };
}
