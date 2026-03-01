/**************************************************************************/
/* reductions.hpp                                                         */
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
     * @class ReductionOps
     * @brief Parallel tensor reduction operations.
     * 
     * These operations aggregate many elements into a single scalar value. 
     * All results are returned as SYCL events as they are computed asynchronously 
     * and stored in the provided scalar tensor.
     */
    class ReductionOps 
    {
        public:
            /**
             * @brief Construct ReductionOps with a reference to the engine.
             * @param e The SushiBLAS engine.
             */
            explicit ReductionOps(Engine& e) : engine_(e) {}

            /** 
             * @brief Sum of all elements.
             * Computes result = sum(t_i).
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event representing task completion.
             */
            sycl::event sum(const Tensor& t, Tensor& result);

            /** 
             * @brief Mean of all elements.
             * Computes result = sum(t_i) / num_elements.
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event mean(const Tensor& t, Tensor& result);

            /** 
             * @brief Maximum value in the tensor.
             * Finds result = max(t_i).
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event max(const Tensor& t, Tensor& result);

            /** 
             * @brief Minimum value in the tensor.
             * Finds result = min(t_i).
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event min(const Tensor& t, Tensor& result);

            /** 
             * @brief Index of the maximum value.
             * @param t Input tensor.
             * @param result Scalar output tensor (stores index as int64_t/float).
             * @return sycl::event.
             */
            sycl::event argmax(const Tensor& t, Tensor& result);

            /** 
             * @brief Index of the minimum value.
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event argmin(const Tensor& t, Tensor& result);

            /** 
             * @brief Variance of all elements.
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event var(const Tensor& t, Tensor& result);

            /** 
             * @brief Standard Deviation of all elements.
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event std(const Tensor& t, Tensor& result);

            /** 
             * @brief Product of all elements.
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event prod(const Tensor& t, Tensor& result);

            /** 
             * @brief Log-Sum-Exp reduction.
             * Computes result = log(sum(exp(t_i - max(t)))).
             * Used for numerical stability in softmax calculations.
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event logsumexp(const Tensor& t, Tensor& result);

            /** 
             * @brief L1 Norm (Sum of absolute values).
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event asum(const Tensor& t, Tensor& result);

            /** 
             * @brief L2 Norm (Euclidean norm).
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return sycl::event.
             */
            sycl::event nrm2(const Tensor& t, Tensor& result);

        private:
            Engine& engine_;
    };

} // namespace SushiBLAS
