/**************************************************************************/
/* nonlinear.hpp                                                          */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file nonlinear.hpp
 * @brief Declares NonLinearOps, the activation functions and their backward passes.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>

namespace SushiBLAS 
{
    /**
     * @class NonLinearOps
     * @brief Records activation functions and their backward passes on contiguous tensors.
     * @note Every operand must have storage and dense strides, and the operands of a backward
     * pass must share one shape: the library does not broadcast. Sigmoid and tanh accept real
     * and complex dtypes; every other activation accepts real dtypes only.
     */
    class NonLinearOps final
    {
        public:
            /** @brief Builds a view that records through recorder, which must outlive the view. */
            explicit NonLinearOps(TaskRecorder& recorder) noexcept : recorder_(recorder) {}

            /** @brief Builds a view that records on engine's own graph. */
            explicit NonLinearOps(Engine& engine) noexcept : NonLinearOps(engine.get_recorder()) {}

            NonLinearOps(const NonLinearOps&) = default;
            NonLinearOps& operator=(const NonLinearOps&) = delete;
            NonLinearOps(NonLinearOps&&) = default;
            NonLinearOps& operator=(NonLinearOps&&) = delete;
            ~NonLinearOps() = default;

            /**
             * @brief Rectified Linear Unit (ReLU).
             * Computes f(x) = max(0, x) element-wise.
             * @param t Input/Output tensor (modified in-place).
             * @return The handle of the recorded task.
             */
            TaskHandle relu(Tensor& t);

            /**
             * @brief ReLU Backward (Gradient).
             * Computes dx = dy * (x > 0 ? 1 : 0).
             * @param dy Output gradient (incoming).
             * @param x Original forward input.
             * @param dx Input gradient result.
             * @return The handle of the recorded task.
             */
            TaskHandle relu_backward(const Tensor& dy, const Tensor& x, Tensor& dx);

            /**
             * @brief Leaky Rectified Linear Unit (LeakyReLU).
             * Computes f(x) = x if x > 0 else alpha * x.
             * @param t Input/Output tensor.
             * @param alpha Small slope for negative values (default: 0.01).
             * @return The handle of the recorded task.
             */
            TaskHandle leaky_relu(Tensor& t, double alpha = 0.01);

            /**
             * @brief Computes dx = dy * (x > 0 ? 1 : alpha), the LeakyReLU backward pass.
             * @param x The forward input.
             */
            TaskHandle leaky_relu_backward(const Tensor& dy, const Tensor& x, Tensor& dx, double alpha = 0.01);

            /**
             * @brief Sigmoid Activation.
             * Computes f(x) = 1 / (1 + exp(-x)).
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle sigmoid(Tensor& t);

            /**
             * @brief Sigmoid Backward.
             * Computes dx = dy * y * (1 - y) where y = sigmoid(x).
             * @param dy Output gradient.
             * @param y Forward output (the sigmoid result).
             * @param dx Gradient result.
             * @return The handle of the recorded task.
             */
            TaskHandle sigmoid_backward(const Tensor& dy, const Tensor& y, Tensor& dx);

            /**
             * @brief Hyperbolic Tangent (Tanh).
             * Computes f(x) = tanh(x).
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle tanh(Tensor& t);

            /**
             * @brief Tanh Backward.
             * Computes dx = dy * (1 - y^2) where y = tanh(x).
             * @param dy Output gradient.
             * @param y Forward output (the tanh result).
             * @param dx Gradient result.
             * @return The handle of the recorded task.
             */
            TaskHandle tanh_backward(const Tensor& dy, const Tensor& y, Tensor& dx);

            /**
             * @brief Exponential Linear Unit (ELU).
             * Computes f(x) = x if x > 0 else alpha * (exp(x) - 1).
             * @param t Input/Output tensor.
             * @param alpha ELU scale factor.
             * @return The handle of the recorded task.
             */
            TaskHandle elu(Tensor& t, double alpha = 1.0);

            /**
             * @brief Computes dx = dy * (x > 0 ? 1 : alpha * exp(x)), the ELU backward pass.
             * @param x The forward input.
             */
            TaskHandle elu_backward(const Tensor& dy, const Tensor& x, Tensor& dx, double alpha = 1.0);

            /**
             * @brief Sigmoid Linear Unit (SiLU / Swish).
             * Computes f(x) = x * sigmoid(x).
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle silu(Tensor& t);

            /**
             * @brief SiLU Backward.
             * Computes dx = dy * (sig(x) * (1 + x * (1 - sig(x)))).
             * @param dy Output gradient.
             * @param x Forward input.
             * @param dx Gradient result.
             * @return The handle of the recorded task.
             */
            TaskHandle silu_backward(const Tensor& dy, const Tensor& x, Tensor& dx);

            /**
             * @brief Gaussian Error Linear Unit (GELU).
             * Computes f(x) = x * P(X <= x) where X ~ N(0, 1).
             * Approximated as: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle gelu(Tensor& t);

            /**
             * @brief GELU Backward.
             * @param dy Output gradient.
             * @param x Forward input.
             * @param dx Gradient result.
             * @return The handle of the recorded task.
             */
            TaskHandle gelu_backward(const Tensor& dy, const Tensor& x, Tensor& dx);

            /**
             * @brief Softplus activation.
             * Computes f(x) = max(x, 0) + log1p(exp(-|x|)), which equals ln(1 + exp(x)) without overflow.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle softplus(Tensor& t);

            /**
             * @brief Softplus Backward.
             * Computes dx = dy * sigmoid(x).
             * @param dy Output gradient.
             * @param x Forward input.
             * @param dx Gradient result.
             * @return The handle of the recorded task.
             */
            TaskHandle softplus_backward(const Tensor& dy, const Tensor& x, Tensor& dx);

        private:
            /// @brief The recorder every operation of this view records through.
            TaskRecorder& recorder_;
    };

} // namespace SushiBLAS
