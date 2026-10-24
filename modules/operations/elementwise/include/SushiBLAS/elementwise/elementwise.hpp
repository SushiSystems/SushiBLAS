/**************************************************************************/
/* elementwise.hpp                                                        */
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
 * @file elementwise.hpp
 * @brief Declares ElementwiseOps, the per-element math operations on tensors.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>

namespace SushiBLAS 
{
    /**
     * @class ElementwiseOps
     * @brief Records element-wise arithmetic and mathematical operations on contiguous tensors.
     * @note Every operand must have storage and dense strides, and the operands of one call must
     * share one shape: the library does not broadcast.
     */
    class ElementwiseOps final
    {
        public:
            /** @brief Builds a view that records through recorder, which must outlive the view. */
            explicit ElementwiseOps(TaskRecorder& recorder) noexcept : recorder_(recorder) {}

            /** @brief Builds a view that records on engine's own graph. */
            explicit ElementwiseOps(Engine& engine) noexcept
                : ElementwiseOps(engine.recorder())
            {

            }

            ElementwiseOps(const ElementwiseOps&) = default;
            ElementwiseOps& operator=(const ElementwiseOps&) = delete;
            ElementwiseOps(ElementwiseOps&&) = default;
            ElementwiseOps& operator=(ElementwiseOps&&) = delete;
            ~ElementwiseOps() = default;

            /** 
             * @brief Element-wise addition.
             * Computes C = A + B element-wise.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param C Output tensor to store the result.
             * @return The handle of the recorded task.
             */
            TaskHandle add(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise subtraction.
             * Computes C = A - B element-wise.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param C Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle sub(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise multiplication (Hadamard product).
             * Computes C = A * B element-wise.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param C Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle mul(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise division.
             * Computes C = A / B element-wise.
             * @param A First input tensor (dividend).
             * @param B Second input tensor (divisor).
             * @param C Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle div(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise square root.
             * Computes t = sqrt(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle sqrt(Tensor& t);

            /** 
             * @brief Element-wise square.
             * Computes t = t * t for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle square(Tensor& t);

            /** 
             * @brief Element-wise reciprocal.
             * Computes t = 1 / t for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle reciprocal(Tensor& t);

            /** 
             * @brief Element-wise negation.
             * Computes t = -t for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle neg(Tensor& t);

            /** 
             * @brief Element-wise exponential.
             * Computes t = exp(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle exp(Tensor& t);

            /** 
             * @brief Element-wise natural logarithm.
             * Computes t = log(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle log(Tensor& t);

            /** 
             * @brief Element-wise absolute value.
             * Computes t = abs(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle abs(Tensor& t);

            /**
             * @brief Raises each element of t to exponent in place.
             * @param exponent The power, narrowed to the element type when the call is recorded.
             * @return The handle of the recorded task.
             */
            TaskHandle pow(Tensor& t, double exponent);

            /** 
             * @brief Element-wise sine.
             * Computes t = sin(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle sin(Tensor& t);

            /** 
             * @brief Element-wise cosine.
             * Computes t = cos(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle cos(Tensor& t);

            /** 
             * @brief Element-wise tangent.
             * Computes t = tan(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle tan(Tensor& t);

            /** 
             * @brief Element-wise arcsine.
             * Computes t = asin(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle asin(Tensor& t);

            /** 
             * @brief Element-wise arccosine.
             * Computes t = acos(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle acos(Tensor& t);

            /** 
             * @brief Element-wise arctangent.
             * Computes t = atan(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle atan(Tensor& t);

            /** 
             * @brief Element-wise hyperbolic sine.
             * Computes t = sinh(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle sinh(Tensor& t);

            /** 
             * @brief Element-wise hyperbolic cosine.
             * Computes t = cosh(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle cosh(Tensor& t);

            /** 
             * @brief Element-wise inverse hyperbolic sine.
             * Computes t = asinh(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle asinh(Tensor& t);

            /** 
             * @brief Element-wise inverse hyperbolic cosine.
             * Computes t = acosh(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle acosh(Tensor& t);

            /** 
             * @brief Element-wise inverse hyperbolic tangent.
             * Computes t = atanh(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle atanh(Tensor& t);

            /** 
             * @brief Element-wise ceiling.
             * Computes t = ceil(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle ceil(Tensor& t);

            /** 
             * @brief Element-wise floor.
             * Computes t = floor(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle floor(Tensor& t);

            /** 
             * @brief Element-wise rounding.
             * Computes t = round(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle round(Tensor& t);

            /** 
             * @brief Element-wise minimum.
             * Computes C = min(A, B) element-wise.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param C Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle min(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise maximum.
             * Computes C = max(A, B) element-wise.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param C Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle max(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise floating-point remainder.
             * Computes C = fmod(A, B) element-wise.
             * @param A Dividend tensor.
             * @param B Divisor tensor.
             * @param C Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle fmod(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise remainder.
             * Computes C = remainder(A, B) element-wise.
             * @param A Dividend tensor.
             * @param B Divisor tensor.
             * @param C Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle remainder(const Tensor& A, const Tensor& B, Tensor& C);

            /**
             * @brief Clamps each element of t in place to `[min_val, max_val]`.
             * @param min_val Lower bound; an integer tensor truncates it toward zero.
             * @param max_val Upper bound; an integer tensor truncates it toward zero.
             * @return The handle of the recorded task.
             * @throws std::invalid_argument if min_val > max_val, when the call is recorded.
             */
            TaskHandle clamp(Tensor& t, double min_val, double max_val);

            /**
             * @brief Copies src into dst, converting the element type.
             * @param src Must be contiguous and have storage.
             * @param dst Must be contiguous, have storage, match src's shape and layout and not overlap it.
             * @return The handle of the recorded task.
             * @throws std::invalid_argument if an operand has no storage, is not contiguous or falls outside Domain::Real, or the shapes or layouts differ.
             * @see modules/operations/MATH_README.md
             */
            TaskHandle convert(const Tensor& src, Tensor& dst);

            /** @brief Holds the most inputs one ordered_sum call takes. */
            static constexpr std::size_t K_MAX_ORDERED_SUM_INPUTS = 32;

            /**
             * @brief Writes out = w0 * x0 + w1 * x1 + ... in index order, in one kernel.
             * @param inputs One to K_MAX_ORDERED_SUM_INPUTS tensors of one shape, layout and type.
             * @param weights One real weight per input; an integer sum accepts only 1.
             * @param out Matches the inputs in shape, layout and type and overlaps none of them.
             * @throws std::invalid_argument on a count, operand, match, overlap or weight rule.
             * @see modules/operations/elementwise/README.md
             */
            TaskHandle ordered_sum(SushiRuntime::span<const Tensor> inputs,
                                   SushiRuntime::span<const double> weights, Tensor& out);

        private:
            /// @brief The recorder every operation of this view records through.
            TaskRecorder& recorder_;
    };
} // namespace SushiBLAS
