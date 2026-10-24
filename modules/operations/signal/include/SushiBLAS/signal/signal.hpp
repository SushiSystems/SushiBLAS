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
             * @brief Convolves every row of signal with taps in valid mode: `out[n] = sum_k signal[n + K - 1 - k] * taps[k]`.
             * @param signal Rank 1 or more; the rows are its last dimension, `L` samples each, batched over the leading ones.
             * @param taps The rank-1 `[K]` taps, with `1 <= K <= L`.
             * @param out Shaped like signal with the last dimension `L - K + 1`; it may share no byte with an input.
             * @throws std::invalid_argument at record time when the ranks, dtypes or extents disagree, an operand is not dense, or out overlaps an input.
             * @see modules/operations/signal/README.md
             */
            TaskHandle convolve_1d(const Tensor& signal, const Tensor& taps, Tensor& out);

        private:
            /// @brief The recorder every operation of this view records through.
            TaskRecorder& recorder_;
    };
} // namespace SushiBLAS
