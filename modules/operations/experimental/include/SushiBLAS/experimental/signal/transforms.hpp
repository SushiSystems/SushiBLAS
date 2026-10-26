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
 * @brief Declares TransformsOps, the batched forward and inverse fast Fourier transforms.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/operation_options.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>

namespace SushiBLAS
{
    /** @brief Names the element pairing of a transform's operands. */
    enum class FftKind : std::uint8_t
    {
        ComplexToComplex, /**< Both sides complex. */
        RealToComplex,    /**< Real samples and the columns / 2 + 1 half spectrum per row. */
    };

    /**
     * @brief Holds the extent of one batched transform.
     * @see modules/operations/experimental/README.md
     */
    struct FftShape
    {
            std::size_t rows = 1;    /**< Rows of each transform; 1 for a 1-D transform. */
            std::size_t columns = 0; /**< Samples per row, the innermost and contiguous axis. */
            std::size_t batch = 1;   /**< Transforms laid out one after another. */
            FftKind kind = FftKind::ComplexToComplex; /**< Element pairing of the operands. */
    };

    /**
     * @class TransformsOps
     * @brief Records batched one- and two-dimensional discrete Fourier transforms.
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
             * @brief Records the unscaled forward transform of each batch entry.
             * @param input COMPLEX32 or COMPLEX64, rows * columns per entry; FLOAT32 or FLOAT64
             *        for RealToComplex.
             * @param output The complex dtype of input's scalar, rows * columns elements per entry,
             *        rows * (columns / 2 + 1) for RealToComplex; no byte shared with input.
             * @throws std::invalid_argument at record time on a mismatched or non-dense operand.
             */
            TaskHandle fft(const Tensor& input, Tensor& output, const FftShape& shape);

            /**
             * @brief Records fft(input, output, shape) under the name and class options states.
             * @throws std::invalid_argument as the form without options, and for an empty name.
             */
            TaskHandle fft(const Tensor& input, Tensor& output, const FftShape& shape,
                           const OperationOptions& options);

            /**
             * @brief Records the inverse of fft, scaled by 1 / (rows * columns).
             * @param input The spectrum laid out as fft writes it for shape.
             * @param output The samples laid out as fft reads them; no byte shared with input.
             * @throws std::invalid_argument at record time on a mismatched or non-dense operand.
             */
            TaskHandle inverse_fft(const Tensor& input, Tensor& output, const FftShape& shape);

            /**
             * @brief Records inverse_fft(input, output, shape) under the name and class options
             *        states.
             * @throws std::invalid_argument as the form without options, and for an empty name.
             */
            TaskHandle inverse_fft(const Tensor& input, Tensor& output, const FftShape& shape,
                                   const OperationOptions& options);

        private:
            /// @brief The recorder every operation of this view records through.
            TaskRecorder& recorder_;
    };
} // namespace SushiBLAS
