/**************************************************************************/
/* elementwise.hpp                                                        */
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
     * @class ElementwiseOps
     * @brief Element-wise arithmetic and mathematical operations for tensors.
     * 
     * Provides high-performance independent operations where each element 
     * in the result depends only on its corresponding element in the input(s).
     */
    class ElementwiseOps 
    {
        public:
            /**
            * @brief Construct ElementwiseOps with a reference to the engine.
            * @param e The SushiBLAS engine.
            */
            explicit ElementwiseOps(Engine& e) : engine_(e) {}

            /** 
            * @brief Element-wise addition.
            * Computes C = A + B element-wise.
            * @param A First input tensor.
            * @param B Second input tensor.
            * @param C Output tensor to store the result.
            * @return sycl::event representing task completion.
            */
            sycl::event add(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
            * @brief Element-wise subtraction.
            * Computes C = A - B element-wise.
            * @param A First input tensor.
            * @param B Second input tensor.
            * @param C Output tensor.
            * @return sycl::event.
            */
            sycl::event sub(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
            * @brief Element-wise multiplication (Hadamard product).
            * Computes C = A * B element-wise.
            * @param A First input tensor.
            * @param B Second input tensor.
            * @param C Output tensor.
            * @return sycl::event.
            */
            sycl::event mul(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
            * @brief Element-wise division.
            * Computes C = A / B element-wise.
            * @param A First input tensor (dividend).
            * @param B Second input tensor (divisor).
            * @param C Output tensor.
            * @return sycl::event.
            */
            sycl::event div(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
            * @brief Element-wise square root.
            * Computes t = sqrt(t) for each element in-place.
            * @param t Input/Output tensor.
            * @return sycl::event.
            */
            sycl::event sqrt(Tensor& t);

            /** 
            * @brief Element-wise exponential.
            * Computes t = exp(t) for each element in-place.
            * @param t Input/Output tensor.
            * @return sycl::event.
            */
            sycl::event exp(Tensor& t);

            /** 
            * @brief Element-wise natural logarithm.
            * Computes t = log(t) for each element in-place.
            * @param t Input/Output tensor.
            * @return sycl::event.
            */
            sycl::event log(Tensor& t);

            /** 
            * @brief Element-wise absolute value.
            * Computes t = abs(t) for each element in-place.
            * @param t Input/Output tensor.
            * @return sycl::event.
            */
            sycl::event abs(Tensor& t);

            /** 
            * @brief Element-wise power function.
            * Computes t = t^exponent for each element in-place.
            * @param t Input/Output tensor.
            * @param exponent The power to raise elements to.
            * @return sycl::event.
            */
            sycl::event pow(Tensor& t, float exponent);

        private:
            Engine& engine_;
    };

} // namespace SushiBLAS
