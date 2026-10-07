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

#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor.hpp>

namespace SushiBLAS 
{
    class Engine;

    /**
     * @class ElementwiseOps
     * @brief Records element-wise arithmetic and mathematical operations on contiguous tensors.
     * @note Every operand must have storage and dense strides, and the operands of one call must
     * share one shape: the library does not broadcast.
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
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event add(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise subtraction.
             * Computes C = A - B element-wise.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param C Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event sub(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise multiplication (Hadamard product).
             * Computes C = A * B element-wise.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param C Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event mul(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise division.
             * Computes C = A / B element-wise.
             * @param A First input tensor (dividend).
             * @param B Second input tensor (divisor).
             * @param C Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event div(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise square root.
             * Computes t = sqrt(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event sqrt(Tensor& t);

            /** 
             * @brief Element-wise square.
             * Computes t = t * t for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event square(Tensor& t);

            /** 
             * @brief Element-wise reciprocal.
             * Computes t = 1 / t for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event reciprocal(Tensor& t);

            /** 
             * @brief Element-wise negation.
             * Computes t = -t for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event neg(Tensor& t);

            /** 
             * @brief Element-wise exponential.
             * Computes t = exp(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event exp(Tensor& t);

            /** 
             * @brief Element-wise natural logarithm.
             * Computes t = log(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event log(Tensor& t);

            /** 
             * @brief Element-wise absolute value.
             * Computes t = abs(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event abs(Tensor& t);

            /**
             * @brief Raises each element of t to exponent in place.
             * @param exponent The power, narrowed to the element type when the call is recorded.
             * @return A default-constructed, already-complete event; the task runs at
             * Engine::execute.
             */
            sycl::event pow(Tensor& t, double exponent);

            /** 
             * @brief Element-wise sine.
             * Computes t = sin(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event sin(Tensor& t);

            /** 
             * @brief Element-wise cosine.
             * Computes t = cos(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event cos(Tensor& t);

            /** 
             * @brief Element-wise tangent.
             * Computes t = tan(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event tan(Tensor& t);

            /** 
             * @brief Element-wise arcsine.
             * Computes t = asin(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event asin(Tensor& t);

            /** 
             * @brief Element-wise arccosine.
             * Computes t = acos(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event acos(Tensor& t);

            /** 
             * @brief Element-wise arctangent.
             * Computes t = atan(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event atan(Tensor& t);

            /** 
             * @brief Element-wise hyperbolic sine.
             * Computes t = sinh(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event sinh(Tensor& t);

            /** 
             * @brief Element-wise hyperbolic cosine.
             * Computes t = cosh(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event cosh(Tensor& t);

            /** 
             * @brief Element-wise inverse hyperbolic sine.
             * Computes t = asinh(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event asinh(Tensor& t);

            /** 
             * @brief Element-wise inverse hyperbolic cosine.
             * Computes t = acosh(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event acosh(Tensor& t);

            /** 
             * @brief Element-wise inverse hyperbolic tangent.
             * Computes t = atanh(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event atanh(Tensor& t);

            /** 
             * @brief Element-wise ceiling.
             * Computes t = ceil(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event ceil(Tensor& t);

            /** 
             * @brief Element-wise floor.
             * Computes t = floor(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event floor(Tensor& t);

            /** 
             * @brief Element-wise rounding.
             * Computes t = round(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event round(Tensor& t);

            /** 
             * @brief Element-wise minimum.
             * Computes C = min(A, B) element-wise.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param C Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event min(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise maximum.
             * Computes C = max(A, B) element-wise.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param C Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event max(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise floating-point remainder.
             * Computes C = fmod(A, B) element-wise.
             * @param A Dividend tensor.
             * @param B Divisor tensor.
             * @param C Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event fmod(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise remainder.
             * Computes C = remainder(A, B) element-wise.
             * @param A Dividend tensor.
             * @param B Divisor tensor.
             * @param C Output tensor.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             */
            sycl::event remainder(const Tensor& A, const Tensor& B, Tensor& C);

            /**
             * @brief Clamps each element of t in place to `[min_val, max_val]`.
             * @param min_val Lower bound; an integer tensor truncates it toward zero.
             * @param max_val Upper bound; an integer tensor truncates it toward zero.
             * @return A default-constructed, already-complete event; the task runs at
             * Engine::execute.
             * @throws std::runtime_error if min_val > max_val, when the call is recorded.
             */
            sycl::event clamp(Tensor& t, double min_val, double max_val);

            /**
             * @brief Copies src into dst, converting the element type.
             * @param src Must be contiguous and have storage.
             * @param dst Must be contiguous, have storage, match src's shape and layout and not overlap it.
             * @return A default-constructed, already-complete event; the task runs at Engine::execute.
             * @throws std::invalid_argument if an operand has no storage, is not contiguous, or the shapes or layouts differ.
             * @throws std::runtime_error naming both dtypes if either falls outside Domain::REAL.
             * @see include/SushiBLAS/engine/math/README.md
             */
            sycl::event convert(const Tensor& src, Tensor& dst);

        private:
            Engine& engine_;
    };
} // namespace SushiBLAS
