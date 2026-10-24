/**************************************************************************/
/* signal.hpp                                                             */
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

/**
 * @file signal.hpp
 * @brief Declares SignalOps, the operations over rows of samples.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/operation_options.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>

namespace SushiBLAS
{
    /**
     * @class SignalOps
     * @brief Records operations that act on each row of a signal on its own.
     */
    class SignalOps final
    {
        public:
            /** @brief Builds a view that records through recorder, which must outlive the view. */
            explicit SignalOps(TaskRecorder& recorder) noexcept : recorder_(recorder) {}

            SignalOps(const SignalOps&) = default;
            SignalOps& operator=(const SignalOps&) = delete;
            SignalOps(SignalOps&&) = default;
            SignalOps& operator=(SignalOps&&) = delete;
            ~SignalOps() = default;

            /**
             * @brief Convolves each row of signal with taps in valid mode; see the module README.
             * @param signal Rank 1 or more; the rows are its last dimension, `L` samples each.
             * @param taps The rank-1 `[K]` taps, with `1 <= K <= L`.
             * @param out Like signal, last dimension `L - K + 1`; no byte shared with an input.
             * @throws std::invalid_argument at record time on a mismatched or non-dense operand.
             * @see modules/operations/signal/README.md
             */
            TaskHandle convolve_1d(const Tensor& signal, const Tensor& taps, Tensor& out);

            /**
             * @brief Records the convolution as convolve_1d(signal, taps, out) does, under the
             *        name and the determinism class options states.
             * @param options The label the node reports under and the contract it carries to the
             *        plan of the runtime.
             * @throws std::invalid_argument as the form without options, and for an empty name or a
             *         class outside the enum.
             */
            TaskHandle convolve_1d(const Tensor& signal, const Tensor& taps, Tensor& out,
                                   const OperationOptions& options);

        private:
            /// @brief The recorder every operation of this view records through.
            TaskRecorder& recorder_;
    };
} // namespace SushiBLAS
