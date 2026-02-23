/**************************************************************************/
/* logic.hpp                                                              */
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
     * @class LogicOps
     * @brief Logical and comparison operations for tensors.
     * 
     * These operations are essential for masking, filtering, and 
     * implementing conditional logic in computational graphs.
     */
    class LogicOps 
    {
        public:
            explicit LogicOps(Engine& e) : engine_(e) {}

            /**
             * @brief Element-wise comparison: A > B.
             * @param A First tensor.
             * @param B Second tensor.
             * @param result Boolean/Integer mask tensor.
             * @return sycl::event.
             */
            sycl::event greater(const Tensor& A, const Tensor& B, Tensor& result);

            /**
             * @brief Element-wise comparison: A < B.
             * @param A First tensor.
             * @param B Second tensor.
             * @param result Boolean mask tensor.
             * @return sycl::event.
             */
            sycl::event less(const Tensor& A, const Tensor& B, Tensor& result);

            /**
             * @brief Conditional selection (Like numpy.where).
             * Computes: out = condition ? A : B
             * @param condition Mask tensor (non-zero means true).
             * @param A Selection if true.
             * @param B Selection if false.
             * @param out Output tensor.
             * @return sycl::event.
             */
            sycl::event where(const Tensor& condition, const Tensor& A, const Tensor& B, Tensor& out);

            /**
             * @brief Logical AND of two boolean masks.
             * @param A Input mask A.
             * @param B Input mask B.
             * @param out Output mask.
             * @return sycl::event.
             */
            sycl::event logical_and(const Tensor& A, const Tensor& B, Tensor& out);

            /**
             * @brief Check if all elements are non-zero (Logical ALL).
             * @param t Input tensor.
             * @param result Scalar output (1 if all true, else 0).
             * @return sycl::event.
             */
            sycl::event all(const Tensor& t, Tensor& result);

            /**
             * @brief Check if any element is non-zero (Logical ANY).
             * @param t Input tensor.
             * @param result Scalar output.
             * @return sycl::event.
             */
            sycl::event any(const Tensor& t, Tensor& result);

        private:
            Engine& engine_;
    };

} // namespace SushiBLAS
