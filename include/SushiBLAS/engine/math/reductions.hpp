/**************************************************************************/
/* reductions.hpp                                                         */
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
 * @file reductions.hpp
 * @brief Declares ReductionOps, the whole-tensor and per-axis reductions.
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
     * @class ReductionOps
     * @brief Records reductions of a tensor into a scalar result tensor or along an axis.
     * @note A whole-tensor reduction needs a contiguous operand with storage. The axis
     * reductions honour strides. Var, std, logsumexp, argmax and argmin reject an empty tensor.
     */
    class ReductionOps final
    {
        public:
            /** @brief Builds a view that records through recorder, which must outlive the view. */
            explicit ReductionOps(TaskRecorder& recorder) noexcept : recorder_(recorder) {}

            /** @brief Builds a view that records on engine's own graph. */
            explicit ReductionOps(Engine& engine) noexcept : ReductionOps(engine.get_recorder()) {}

            ReductionOps(const ReductionOps&) = default;
            ReductionOps& operator=(const ReductionOps&) = delete;
            ReductionOps(ReductionOps&&) = default;
            ReductionOps& operator=(ReductionOps&&) = delete;
            ~ReductionOps() = default;

            /** 
             * @brief Sum of all elements.
             * Computes result = sum(t_i).
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle sum(const Tensor& t, Tensor& result);

            /** 
             * @brief Mean of all elements.
             * Computes result = sum(t_i) / num_elements.
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle mean(const Tensor& t, Tensor& result);

            /** 
             * @brief Maximum value in the tensor.
             * Finds result = max(t_i). Any NaN element makes the result NaN; an empty tensor
             * gives std::numeric_limits<T>::lowest().
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle max(const Tensor& t, Tensor& result);

            /** 
             * @brief Minimum value in the tensor.
             * Finds result = min(t_i). Any NaN element makes the result NaN; an empty tensor
             * gives std::numeric_limits<T>::max().
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle min(const Tensor& t, Tensor& result);

            /** 
             * @brief Index of the maximum value.
             * @param t Input tensor.
             * @param result Scalar INT64 tensor; receives the lowest index of the maximum.
             * @throws std::invalid_argument if t is empty.
             * @return The handle of the recorded task.
             */
            TaskHandle argmax(const Tensor& t, Tensor& result);

            /** 
             * @brief Index of the minimum value.
             * @param t Input tensor.
             * @param result Scalar INT64 tensor; receives the lowest index of the minimum.
             * @throws std::invalid_argument if t is empty.
             * @return The handle of the recorded task.
             */
            TaskHandle argmin(const Tensor& t, Tensor& result);

            /** 
             * @brief Variance of all elements.
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @throws std::invalid_argument if t is empty.
             * @return The handle of the recorded task.
             */
            TaskHandle var(const Tensor& t, Tensor& result);

            /** 
             * @brief Standard Deviation of all elements.
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @throws std::invalid_argument if t is empty.
             * @return The handle of the recorded task.
             */
            TaskHandle std(const Tensor& t, Tensor& result);

            /** 
             * @brief Product of all elements.
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle prod(const Tensor& t, Tensor& result);

            /** 
             * @brief Log-Sum-Exp reduction.
             * Computes result = log(sum(exp(t_i - max(t)))).
             * Used for numerical stability in softmax calculations.
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @throws std::invalid_argument if t is empty.
             * @return The handle of the recorded task.
             */
            TaskHandle logsumexp(const Tensor& t, Tensor& result);

            /**
             * @brief Computes the maximum along one axis of a rank-2 tensor.
             * @param axis 0 for a per-column maximum, 1 for a per-row maximum.
             * @param result Rank-1 tensor of the surviving length, sharing t's dtype.
             * @throws std::runtime_error on a rank, axis, length or dtype mismatch.
             * @see include/SushiBLAS/engine/math/README.md
             */
            TaskHandle max_along(const Tensor& t, int axis, Tensor& result);

            /**
             * @brief Computes the Neumaier-compensated sum along one axis of a rank-2 tensor.
             * @param axis 0 for column sums, 1 for row sums.
             * @param result Rank-1 tensor of the surviving length, sharing t's dtype.
             * @throws std::runtime_error on a rank, axis, length or dtype mismatch.
             * @see include/SushiBLAS/engine/math/README.md
             */
            TaskHandle sum_along(const Tensor& t, int axis, Tensor& result);

            /**
             * @brief Computes the mean over the two spatial axes of an NHWC tensor: `[N, H, W, C] -> [N, C]`.
             * @param t Must be rank 4, NHWC, with rows contiguous with one another.
             * @param result Rank-2 `[N, C]` tensor sharing t's dtype.
             * @throws std::runtime_error on a rank, shape, stride or dtype mismatch.
             * @see include/SushiBLAS/engine/math/README.md
             */
            TaskHandle mean_spatial(const Tensor& t, Tensor& result);

            /**
             * @brief L1 Norm (Sum of absolute values).
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle asum(const Tensor& t, Tensor& result);

            /**
             * @brief L2 Norm (Euclidean norm).
             * @param t Input tensor.
             * @param result Scalar output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle nrm2(const Tensor& t, Tensor& result);

            /**
             * @brief Counts the elements of t that are infinite or NaN into an INT64 scalar.
             * @param t Must be HALF, FLOAT32 or FLOAT64.
             * @param result Must hold exactly one element of dtype INT64.
             * @throws std::runtime_error if result is not an INT64 scalar or t's dtype is not real.
             * @see include/SushiBLAS/engine/math/README.md
             */
            TaskHandle count_nonfinite(const Tensor& t, Tensor& result);

        private:
            /// @brief The recorder every operation of this view records through.
            TaskRecorder& recorder_;
    };

} // namespace SushiBLAS
