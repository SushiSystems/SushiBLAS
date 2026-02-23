/**************************************************************************/
/* nonlinear.hpp                                                          */
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
     * @class NonLinearOps
     * @brief Non-linear element-wise operations and activation functions.
     * 
     * This class provides high-performance implementations of common activation 
     * functions used in neural networks and general non-linear transformations.
     * Each operation includes both forward and backward (gradient) functions.
     */
    class NonLinearOps 
    {
        public:
            /**
             * @brief Construct NonLinearOps with a reference to the engine.
             * @param e The SushiBLAS engine.
             */
            explicit NonLinearOps(Engine& e) : engine_(e) {}

            /**
             * @brief Rectified Linear Unit (ReLU).
             * Computes f(x) = max(0, x) element-wise.
             * @param t Input/Output tensor (modified in-place).
             * @return sycl::event representing task completion.
             */
            sycl::event relu(Tensor& t);

            /**
             * @brief ReLU Backward (Gradient).
             * Computes dx = dy * (x > 0 ? 1 : 0).
             * @param dy Output gradient (incoming).
             * @param x Original forward input.
             * @param dx Input gradient result.
             * @return sycl::event representing task completion.
             */
            sycl::event relu_backward(const Tensor& dy, const Tensor& x, Tensor& dx);

            /**
             * @brief Leaky Rectified Linear Unit (LeakyReLU).
             * Computes f(x) = x if x > 0 else alpha * x.
             * @param t Input/Output tensor.
             * @param alpha Small slope for negative values (default: 0.01).
             * @return sycl::event.
             */
            sycl::event leaky_relu(Tensor& t, float alpha = 0.01f);

            /**
             * @brief LeakyReLU Backward.
             * Computes dx = dy * (x > 0 ? 1 : alpha).
             * @param dy Output gradient.
             * @param x Forward input.
             * @param dx Gradient result.
             * @param alpha Slope for negative values.
             * @return sycl::event.
             */
            sycl::event leaky_relu_backward(const Tensor& dy, const Tensor& x, Tensor& dx, float alpha = 0.01f);

            /**
             * @brief Sigmoid Activation.
             * Computes f(x) = 1 / (1 + exp(-x)).
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event sigmoid(Tensor& t);

            /**
             * @brief Sigmoid Backward.
             * Computes dx = dy * y * (1 - y) where y = sigmoid(x).
             * @param dy Output gradient.
             * @param y Forward output (the sigmoid result).
             * @param dx Gradient result.
             * @return sycl::event.
             */
            sycl::event sigmoid_backward(const Tensor& dy, const Tensor& y, Tensor& dx);

            /**
             * @brief Hyperbolic Tangent (Tanh).
             * Computes f(x) = tanh(x).
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event tanh(Tensor& t);

            /**
             * @brief Tanh Backward.
             * Computes dx = dy * (1 - y^2) where y = tanh(x).
             * @param dy Output gradient.
             * @param y Forward output (the tanh result).
             * @param dx Gradient result.
             * @return sycl::event.
             */
            sycl::event tanh_backward(const Tensor& dy, const Tensor& y, Tensor& dx);

            /**
             * @brief Exponential Linear Unit (ELU).
             * Computes f(x) = x if x > 0 else alpha * (exp(x) - 1).
             * @param t Input/Output tensor.
             * @param alpha ELU scale factor.
             * @return sycl::event.
             */
            sycl::event elu(Tensor& t, float alpha = 1.0f);

            /**
             * @brief ELU Backward.
             * Computes dx = dy * (x > 0 ? 1 : alpha * exp(x)).
             * @param dy Output gradient.
             * @param x Forward input.
             * @param dx Gradient result.
             * @param alpha scale factor.
             * @return sycl::event.
             */
            sycl::event elu_backward(const Tensor& dy, const Tensor& x, Tensor& dx, float alpha = 1.0f);

            /**
             * @brief Sigmoid Linear Unit (SiLU / Swish).
             * Computes f(x) = x * sigmoid(x).
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event silu(Tensor& t);

            /**
             * @brief SiLU Backward.
             * Computes dx = dy * (sig(x) * (1 + x * (1 - sig(x)))).
             * @param dy Output gradient.
             * @param x Forward input.
             * @param dx Gradient result.
             * @return sycl::event.
             */
            sycl::event silu_backward(const Tensor& dy, const Tensor& x, Tensor& dx);

            /**
             * @brief Gaussian Error Linear Unit (GELU).
             * Computes f(x) = x * P(X <= x) where X ~ N(0, 1).
             * Approximated as: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event gelu(Tensor& t);

            /**
             * @brief GELU Backward.
             * @param dy Output gradient.
             * @param x Forward input.
             * @param dx Gradient result.
             * @return sycl::event.
             */
            sycl::event gelu_backward(const Tensor& dy, const Tensor& x, Tensor& dx);

            /**
             * @brief Softplus activation.
             * Computes f(x) = ln(1 + exp(x)).
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event softplus(Tensor& t);

            /**
             * @brief Softplus Backward.
             * Computes dx = dy * sigmoid(x).
             * @param dy Output gradient.
             * @param x Forward input.
             * @param dx Gradient result.
             * @return sycl::event.
             */
            sycl::event softplus_backward(const Tensor& dy, const Tensor& x, Tensor& dx);

        private:
            Engine& engine_;
    };

} // namespace SushiBLAS
