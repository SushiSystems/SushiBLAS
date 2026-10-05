/****************************************************************/
/* sparse.hpp                                                   */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#pragma once

#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor.hpp>

namespace SushiBLAS 
{
    class Engine;

    /**
     * @class SparseBLAS
     * @brief Sparse matrix operations.
     * 
     * This module handles matrices where most elements are zero.
     * It uses specialized storage formats like CSR, CSC, and COO.
     */
    class SparseBLAS 
    {
        public:
            explicit SparseBLAS(Engine& e) : engine_(e) {}

            // TODO: Define a SparseTensor structure to hold values, row_indices, and col_indices.
            
            // TODO: Implement SpMV (Sparse Matrix-Vector Multiplication).
            // Computes y = alpha * A * x + beta * y, where A is sparse.

            // TODO: Implement SpMM (Sparse Matrix-Matrix Multiplication).
            // Computes C = alpha * A * B + beta * C, where A is sparse.

            // TODO: Support different sparse formats (CSR is high priority for Intel mkl).
            
            // TODO: Add support for Sparse-Sparse operations if needed.

        private:
            // Not yet read anywhere - every method here is still a TODO stub.
            [[maybe_unused]] Engine& engine_;
    };

} // namespace SushiBLAS
