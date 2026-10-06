/**************************************************************************/
/* window.hpp                                                             */
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
 * @file window.hpp
 * @brief Declares the builders of a Spatial::WindowSpan and a Spatial::ColumnSpan from tensors.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/spatial/geometry.hpp>

namespace SushiBLAS
{
    namespace Spatial
    {
        /**
         * @brief Describes a windowed operation whose output extents the caller states.
         * @return The span in element counts; the `out_*_stride` fields stay zero.
         * @throws std::runtime_error when in is not rank 4 or a window origin falls outside the padded input.
         * @see include/SushiBLAS/engine/spatial/README.md
         */
        [[nodiscard]] inline WindowSpan window_span_for(const Tensor& in, const ConvGeometry& g,
                                                        int64_t out_h, int64_t out_w,
                                                        const char* name)
        {
            SB_THROW_IF(in.rank != 4, "Spatial operation '{}' needs a rank-4 NHWC input; got rank {}.",
                        name, in.rank);

            WindowSpan span;
            span.batch = in.shape[0];
            span.in_h = in.shape[1];
            span.in_w = in.shape[2];
            span.channels = in.shape[3];
            span.out_h = out_h;
            span.out_w = out_w;

            span.in_batch_stride = in.strides[0];
            span.in_row_stride = in.strides[1];
            span.in_col_stride = in.strides[2];
            span.in_channel_stride = in.strides[3];

            span.geometry = g;

            // A window that starts past the last input row is refused; see include/SushiBLAS/engine/spatial/README.md.
            SB_THROW_IF(out_h > 0 && (out_h - 1) * g.stride_h - g.pad_top >= span.in_h,
                        "Spatial operation '{}' asks for {} output rows, but window {} starts at "
                        "input row {}, which is outside a {}-row input padded by {} above.",
                        name, out_h, out_h - 1, (out_h - 1) * g.stride_h - g.pad_top,
                        span.in_h, g.pad_top);
            SB_THROW_IF(out_w > 0 && (out_w - 1) * g.stride_w - g.pad_left >= span.in_w,
                        "Spatial operation '{}' asks for {} output columns, but window {} starts at "
                        "input column {}, which is outside a {}-column input padded by {} on the left.",
                        name, out_w, out_w - 1, (out_w - 1) * g.stride_w - g.pad_left,
                        span.in_w, g.pad_left);

            return span;
        }

        /**
         * @brief Describes a windowed operation from its two rank-4 NHWC operands.
         * @return The span in element counts, with both operands' strides filled in.
         * @throws std::runtime_error on a rank other than 4, a batch or channel mismatch, or a bad origin.
         * @see include/SushiBLAS/engine/spatial/README.md
         */
        [[nodiscard]] inline WindowSpan window_span_for(const Tensor& in, const ConvGeometry& g,
                                                        const Tensor& out, const char* name)
        {
            SB_THROW_IF(out.rank != 4, "Spatial operation '{}' needs a rank-4 NHWC output; got rank {}.",
                        name, out.rank);

            WindowSpan span = window_span_for(in, g, out.shape[1], out.shape[2], name);

            SB_THROW_IF(out.shape[0] != span.batch,
                        "Spatial operation '{}' has {} input images and {} output images.",
                        name, span.batch, out.shape[0]);
            SB_THROW_IF(out.shape[3] != span.channels,
                        "Spatial operation '{}' has {} input channels and {} output channels.",
                        name, span.channels, out.shape[3]);

            span.out_batch_stride = out.strides[0];
            span.out_row_stride = out.strides[1];
            span.out_col_stride = out.strides[2];
            span.out_channel_stride = out.strides[3];

            return span;
        }

        /**
         * @brief Describes an im2col matrix's addressing and checks its extents.
         * @throws std::runtime_error when cols is not rank 2 or not `span.windows()` by `span.patch_elements()`.
         */
        [[nodiscard]] inline ColumnSpan column_span_for(const Tensor& cols, const WindowSpan& span,
                                                        const char* name)
        {
            SB_THROW_IF(cols.rank != 2,
                        "Spatial operation '{}' needs a rank-2 column matrix; got rank {}.",
                        name, cols.rank);
            SB_THROW_IF(cols.shape[0] != span.windows(),
                        "Spatial operation '{}' produces {} windows but the column matrix has {} rows.",
                        name, span.windows(), cols.shape[0]);
            SB_THROW_IF(cols.shape[1] != span.patch_elements(),
                        "Spatial operation '{}' produces patches of {} elements "
                        "(kernel_h {} * kernel_w {} * channels {}) but the column matrix is {} wide.",
                        name, span.patch_elements(), span.geometry.kernel_h,
                        span.geometry.kernel_w, span.channels, cols.shape[1]);

            return ColumnSpan{cols.strides[0], cols.strides[1]};
        }

    } // namespace Spatial
} // namespace SushiBLAS
