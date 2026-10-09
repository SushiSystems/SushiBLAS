/**************************************************************************/
/* transforms.hpp                                                         */
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
 * @file transforms.hpp
 * @brief Declares TransformsOps, the experimental signal transforms that have no implementation yet.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>

namespace SushiBLAS
{
    /**
     * @class TransformsOps
     * @brief Signal transformation operations (FFT, etc.).
     *
     * Provides interfaces for Fast Fourier Transforms (FFT) in multiple dimensions.
     * Uses oneMKL's high-performance FFT domain underneath.
     * @see modules/operations/experimental/README.md
     */
    class TransformsOps final
    {
        public:
            /** @brief Builds a view that records through recorder, which must outlive the view. */
            explicit TransformsOps(TaskRecorder& recorder) noexcept : recorder_(recorder) {}

            TransformsOps(const TransformsOps&) = default;
            TransformsOps& operator=(const TransformsOps&) = delete;
            TransformsOps(TransformsOps&&) = default;
            TransformsOps& operator=(TransformsOps&&) = delete;
            ~TransformsOps() = default;

            /**
             * @brief 1D Forward Fast Fourier Transform.
             * @param in Input tensor (time domain).
             * @param out Output tensor (frequency domain, complex).
             * @return The handle of the recorded task.
             */
            TaskHandle fft1d(const Tensor& in, Tensor& out);

            /**
             * @brief 1D Inverse Fast Fourier Transform.
             * @param in Input tensor (frequency domain).
             * @param out Output tensor (time domain).
             * @return The handle of the recorded task.
             */
            TaskHandle ifft1d(const Tensor& in, Tensor& out);

            /**
             * @brief 2D Forward Fast Fourier Transform (Gratings/Images).
             * @param in Input tensor.
             * @param out Output complex tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle fft2d(const Tensor& in, Tensor& out);

            /**
             * @brief 2D Inverse Fast Fourier Transform.
             * @param in Input complex tensor.
             * @param out Output tensor.
             * @return The handle of the recorded task.
             */
            TaskHandle ifft2d(const Tensor& in, Tensor& out);

        private:
            /// @brief The recorder every operation of this view records through.
            TaskRecorder& recorder_;
    };

} // namespace SushiBLAS
