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
             * @brief Element-wise equality comparison: A == B.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param result Output tensor (1 if true, 0 if false).
             * @return sycl::event representing the operation.
             */
            sycl::event equal(const Tensor& A, const Tensor& B, Tensor& result);

            /**
             * @brief Element-wise inequality comparison: A != B.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param result Output tensor (1 if true, 0 if false).
             * @return sycl::event representing the operation.
             */
            sycl::event not_equal(const Tensor& A, const Tensor& B, Tensor& result);

            /**
             * @brief Element-wise comparison: A > B.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param result Output tensor (1 if true, 0 if false).
             * @return sycl::event representing the operation.
             */
            sycl::event greater(const Tensor& A, const Tensor& B, Tensor& result);

            /**
             * @brief Element-wise comparison: A < B.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param result Output tensor (1 if true, 0 if false).
             * @return sycl::event representing the operation.
             */
            sycl::event less(const Tensor& A, const Tensor& B, Tensor& result);

            /**
             * @brief Element-wise comparison: A >= B.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param result Output tensor (1 if true, 0 if false).
             * @return sycl::event representing the operation.
             */
            sycl::event greater_equal(const Tensor& A, const Tensor& B, Tensor& result);

            /**
             * @brief Element-wise comparison: A <= B.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param result Output tensor (1 if true, 0 if false).
             * @return sycl::event representing the operation.
             */
            sycl::event less_equal(const Tensor& A, const Tensor& B, Tensor& result);

            /**
             * @brief Conditional selection (Like numpy.where).
             * Computes: out = condition ? A : B
             * @param condition Mask tensor (non-zero treated as true).
             * @param A Selection if true.
             * @param B Selection if false.
             * @param out Output tensor.
             * @return sycl::event representing the operation.
             */
            sycl::event where(const Tensor& condition, const Tensor& A, const Tensor& B, Tensor& out);

            /**
             * @brief Element-wise logical AND of two tensors.
             * @param A Input tensor A.
             * @param B Input tensor B.
             * @param out Output tensor (1 if both non-zero, else 0).
             * @return sycl::event representing the operation.
             */
            sycl::event logical_and(const Tensor& A, const Tensor& B, Tensor& out);

            /**
             * @brief Element-wise logical OR of two tensors.
             * @param A Input tensor A.
             * @param B Input tensor B.
             * @param out Output tensor (1 if either non-zero, else 0).
             * @return sycl::event representing the operation.
             */
            sycl::event logical_or(const Tensor& A, const Tensor& B, Tensor& out);

            /**
             * @brief Element-wise logical XOR of two tensors.
             * @param A Input tensor A.
             * @param B Input tensor B.
             * @param out Output tensor (1 if exactly one is non-zero, else 0).
             * @return sycl::event representing the operation.
             */
            sycl::event logical_xor(const Tensor& A, const Tensor& B, Tensor& out);

            /**
             * @brief Element-wise logical NOT of a tensor.
             * @param A Input tensor.
             * @param out Output tensor (1 if zero, 0 if non-zero).
             * @return sycl::event representing the operation.
             */
            sycl::event logical_not(const Tensor& A, Tensor& out);

            /**
             * @brief Check if all elements are non-zero (Logical ALL reduction).
             * @param t Input tensor.
             * @param result Scalar output (1 if all true, else 0).
             * @return sycl::event representing the operation.
             */
            sycl::event all(const Tensor& t, Tensor& result);

            /**
             * @brief Check if any element is non-zero (Logical ANY reduction).
             * @param t Input tensor.
             * @param result Scalar output (1 if any true, else 0).
             * @return sycl::event representing the operation.
             */
            sycl::event any(const Tensor& t, Tensor& result);

        private:
            Engine& engine_;
    };

} // namespace SushiBLAS
