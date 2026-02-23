/**************************************************************************/
/* sparse.hpp                                                             */
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
            Engine& engine_;
    };

} // namespace SushiBLAS
