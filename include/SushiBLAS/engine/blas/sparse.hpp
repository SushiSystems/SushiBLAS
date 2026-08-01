/**************************************************************************/
/* sparse.hpp                                                             */
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
