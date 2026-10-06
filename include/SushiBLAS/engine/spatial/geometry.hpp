/**************************************************************************/
/* geometry.hpp                                                           */
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
 * @file geometry.hpp
 * @brief Declares the window geometry of a spatial operation and the extents it implies.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiBLAS
{
    namespace Spatial
    {
        /**
         * @brief Holds a 2-D sliding window: its size, step, spacing and origin.
         * @see include/SushiBLAS/engine/spatial/README.md
         */
        struct ConvGeometry
        {
            /** @brief Window height, in input rows. */
            int64_t kernel_h = 1;
            /** @brief Window width, in input columns. */
            int64_t kernel_w = 1;
            /** @brief Rows between the origins of vertically adjacent windows. */
            int64_t stride_h = 1;
            /** @brief Columns between the origins of horizontally adjacent windows. */
            int64_t stride_w = 1;
            /** @brief Rows between vertically adjacent taps of one window. */
            int64_t dilation_h = 1;
            /** @brief Columns between horizontally adjacent taps of one window. */
            int64_t dilation_w = 1;
            /** @brief Implicit zero rows above row 0. */
            int64_t pad_top = 0;
            /** @brief Implicit zero columns left of column 0. */
            int64_t pad_left = 0;
        };

        /**
         * @brief Returns how many windows fit along one axis, or 0 when none fits.
         * @param pad_before Implicit zeros before index 0.
         * @param pad_after Implicit zeros after the last index.
         * @param dilation Spacing between taps; 1 for an undilated window.
         * @see include/SushiBLAS/engine/spatial/README.md
         */
        [[nodiscard]] constexpr int64_t output_extent(int64_t in,
                                                      int64_t pad_before, int64_t pad_after,
                                                      int64_t dilation, int64_t kernel,
                                                      int64_t stride) noexcept
        {
            if (stride <= 0 || kernel <= 0 || dilation <= 0) return 0;

            const int64_t covered = dilation * (kernel - 1) + 1;
            const int64_t padded = in + pad_before + pad_after;
            if (padded < covered) return 0;

            return (padded - covered) / stride + 1;
        }

        /**
         * @brief Holds where one spatial operation's input and output live, in element counts.
         * @see include/SushiBLAS/engine/spatial/README.md
         */
        struct WindowSpan
        {
            /** @brief How many independent images. */
            int64_t batch = 0;
            /** @brief The input's row count. */
            int64_t in_h = 0;
            /** @brief The input's column count. */
            int64_t in_w = 0;
            /** @brief The channel count, shared by input and output. */
            int64_t channels = 0;
            /** @brief The output's row count; @ref output_extent's answer. */
            int64_t out_h = 0;
            /** @brief The output's column count; @ref output_extent's answer. */
            int64_t out_w = 0;

            /** @brief Input elements between consecutive images. */
            int64_t in_batch_stride = 0;
            /** @brief Input elements between vertically adjacent pixels. */
            int64_t in_row_stride = 0;
            /** @brief Input elements between horizontally adjacent pixels. */
            int64_t in_col_stride = 0;
            /** @brief Input elements between adjacent channels of one pixel. */
            int64_t in_channel_stride = 1;

            /** @brief Output elements between consecutive images. */
            int64_t out_batch_stride = 0;
            /** @brief Output elements between vertically adjacent pixels. */
            int64_t out_row_stride = 0;
            /** @brief Output elements between horizontally adjacent pixels. */
            int64_t out_col_stride = 0;
            /** @brief Output elements between adjacent channels of one pixel. */
            int64_t out_channel_stride = 1;

            /** @brief The window itself. */
            ConvGeometry geometry{};

            /** @brief How many windows the operation visits: `batch * out_h * out_w`. */
            [[nodiscard]] constexpr int64_t windows() const noexcept
            {
                return batch * out_h * out_w;
            }

            /** @brief How many elements one window contributes: `kernel_h * kernel_w * channels`. */
            [[nodiscard]] constexpr int64_t patch_elements() const noexcept
            {
                return geometry.kernel_h * geometry.kernel_w * channels;
            }

            /** @brief How many elements an im2col matrix holds: `windows() * patch_elements()`. */
            [[nodiscard]] constexpr int64_t column_elements() const noexcept
            {
                return windows() * patch_elements();
            }

            /** @brief How many elements the input holds: `batch * in_h * in_w * channels`. */
            [[nodiscard]] constexpr int64_t input_elements() const noexcept
            {
                return batch * in_h * in_w * channels;
            }

            /** @brief How many elements the output holds: `windows() * channels`. */
            [[nodiscard]] constexpr int64_t output_elements() const noexcept
            {
                return windows() * channels;
            }
        };

        /**
         * @brief Holds how an im2col matrix is addressed: a row stride and an element stride.
         * @see include/SushiBLAS/engine/spatial/README.md
         */
        struct ColumnSpan
        {
            /** @brief Elements between the starts of consecutive rows. */
            int64_t row_stride = 0;
            /** @brief Elements between consecutive entries of one row. */
            int64_t element_stride = 1;
        };

    } // namespace Spatial
} // namespace SushiBLAS
