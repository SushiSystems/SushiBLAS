/****************************************************************/
/* window.hpp                                                   */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#pragma once

/**
 * @file window.hpp
 * @brief Building a @ref SushiBLAS::Spatial::WindowSpan from tensors.
 *
 * The single place a layout is interpreted. Everything downstream of this
 * function sees element counts, so adding NCHW later is a second stride
 * assignment here and nothing else anywhere.
 */

#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/spatial/geometry.hpp>

namespace SushiBLAS
{
    namespace Spatial
    {
        /**
         * @brief Describes a windowed operation whose output extents are stated.
         *
         * The form `im2col` and `col2im` take: their other operand is a rank-2
         * column matrix, so there is no output image to read `H_o` and `W_o`
         * off, and the caller states them.
         *
         * Reads the strides off @p in rather than deriving them from its shape,
         * so a view with a non-unit stride - a channel slice, a
         * `storage_offset` into a shared allocation - is addressed correctly
         * without any caller knowing it was one.
         *
         * **What it can and cannot check.** The extents are taken, not
         * recomputed, because @ref output_extent needs `pad_bottom` and
         * `pad_right` and @ref ConvGeometry deliberately does not carry them
         * (see its documentation). What it can prove is that every window's
         * *origin* lands inside the padded input: an output row whose window
         * starts at or past `in_h` reads nothing but padding, which is always a
         * caller's arithmetic error and never a legitimate request.
         *
         * @param in The rank-4 NHWC input.
         * @param g The window geometry.
         * @param out_h The output's row count.
         * @param out_w The output's column count.
         * @param name The operation's name, for diagnostics.
         * @return The span, in element counts; the `out_*_stride` fields are
         *         left at zero, since there is no output image to describe.
         * @throws std::runtime_error when @p in is not rank 4, or a window
         *         origin falls outside the padded input.
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

            // A window whose origin is at or past the last input row covers only
            // padding. Legal arithmetic, never a legal request - and the shape it
            // produces is plausible enough to survive every other check here.
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
         *
         * The form every operation with an image output takes - pooling,
         * upsampling, the channel copies, the affine. `H_o` and `W_o` come from
         * @p out, which the caller sized with @ref output_extent while the
         * trailing padding it needs was still in scope.
         *
         * @param in The rank-4 NHWC input.
         * @param g The window geometry.
         * @param out The rank-4 NHWC output.
         * @param name The operation's name, for diagnostics.
         * @return The span, in element counts, with both operands' strides filled in.
         * @throws std::runtime_error when a rank is not 4, the batch or channel
         *         counts disagree, or a window origin falls outside the padded input.
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
         * @brief Describes an im2col matrix's addressing, and checks its extents.
         *
         * @param cols The rank-2 column matrix.
         * @param span The window span the matrix must match.
         * @param name The operation's name, for diagnostics.
         * @return The column addressing, in element counts.
         * @throws std::runtime_error when @p cols is not rank 2, or its extents
         *         are not `span.windows()` by `span.patch_elements()`.
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
