/**************************************************************************/
/* logic.hpp                                                              */
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
 * @file logic.hpp
 * @brief Declares LogicOps, the comparison and logical operations on tensors.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>

namespace SushiBLAS 
{
    /**
     * @class LogicOps
     * @brief Records comparisons, boolean combinations and truth tests of tensors.
     * @note Every operand must have storage and dense strides, and the operands of one call must
     * share one shape: the library does not broadcast.
     */
    class LogicOps final
    {
        public:
            /** @brief Builds a view that records through recorder, which must outlive the view. */
            explicit LogicOps(TaskRecorder& recorder) noexcept : recorder_(recorder) {}

            /** @brief Builds a view that records on engine's own graph. */
            explicit LogicOps(Engine& engine) noexcept : LogicOps(engine.get_recorder()) {}

            LogicOps(const LogicOps&) = default;
            LogicOps& operator=(const LogicOps&) = delete;
            LogicOps(LogicOps&&) = default;
            LogicOps& operator=(LogicOps&&) = delete;
            ~LogicOps() = default;

            /**
             * @brief Element-wise equality comparison: A == B.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param result Output tensor (1 if true, 0 if false).
             * @return The handle of the recorded task.
             */
            TaskHandle equal(const Tensor& A, const Tensor& B, Tensor& result);

            /**
             * @brief Element-wise inequality comparison: A != B.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param result Output tensor (1 if true, 0 if false).
             * @return The handle of the recorded task.
             */
            TaskHandle not_equal(const Tensor& A, const Tensor& B, Tensor& result);

            /**
             * @brief Element-wise comparison: A > B.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param result Output tensor (1 if true, 0 if false).
             * @return The handle of the recorded task.
             */
            TaskHandle greater(const Tensor& A, const Tensor& B, Tensor& result);

            /**
             * @brief Element-wise comparison: A < B.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param result Output tensor (1 if true, 0 if false).
             * @return The handle of the recorded task.
             */
            TaskHandle less(const Tensor& A, const Tensor& B, Tensor& result);

            /**
             * @brief Element-wise comparison: A >= B.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param result Output tensor (1 if true, 0 if false).
             * @return The handle of the recorded task.
             */
            TaskHandle greater_equal(const Tensor& A, const Tensor& B, Tensor& result);

            /**
             * @brief Element-wise comparison: A <= B.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param result Output tensor (1 if true, 0 if false).
             * @return The handle of the recorded task.
             */
            TaskHandle less_equal(const Tensor& A, const Tensor& B, Tensor& result);

            /**
             * @brief Selects A where condition is non-zero and B elsewhere, like numpy.where.
             */
            TaskHandle where(const Tensor& condition, const Tensor& A, const Tensor& B, Tensor& out);

            /**
             * @brief Element-wise logical AND of two tensors.
             * @param A Input tensor A.
             * @param B Input tensor B.
             * @param out Output tensor (1 if both non-zero, else 0).
             * @return The handle of the recorded task.
             */
            TaskHandle logical_and(const Tensor& A, const Tensor& B, Tensor& out);

            /**
             * @brief Element-wise logical OR of two tensors.
             * @param A Input tensor A.
             * @param B Input tensor B.
             * @param out Output tensor (1 if either non-zero, else 0).
             * @return The handle of the recorded task.
             */
            TaskHandle logical_or(const Tensor& A, const Tensor& B, Tensor& out);

            /**
             * @brief Element-wise logical XOR of two tensors.
             * @param A Input tensor A.
             * @param B Input tensor B.
             * @param out Output tensor (1 if exactly one is non-zero, else 0).
             * @return The handle of the recorded task.
             */
            TaskHandle logical_xor(const Tensor& A, const Tensor& B, Tensor& out);

            /**
             * @brief Element-wise logical NOT of a tensor.
             * @param A Input tensor.
             * @param out Output tensor (1 if zero, 0 if non-zero).
             * @return The handle of the recorded task.
             */
            TaskHandle logical_not(const Tensor& A, Tensor& out);

            /**
             * @brief Check if all elements are non-zero (Logical ALL reduction).
             * @param t Input tensor.
             * @param result Scalar output (1 if all true, else 0).
             * @return The handle of the recorded task.
             */
            TaskHandle all(const Tensor& t, Tensor& result);

            /**
             * @brief Check if any element is non-zero (Logical ANY reduction).
             * @param t Input tensor.
             * @param result Scalar output (1 if any true, else 0).
             * @return The handle of the recorded task.
             */
            TaskHandle any(const Tensor& t, Tensor& result);

        private:
            /// @brief The recorder every operation of this view records through.
            TaskRecorder& recorder_;
    };

} // namespace SushiBLAS
