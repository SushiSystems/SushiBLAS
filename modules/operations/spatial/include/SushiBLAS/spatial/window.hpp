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
 * @brief Declares the builders of the Spatial window, column and tap spans from tensors.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/support/spatial_geometry.hpp>

namespace SushiBLAS
{
    namespace Spatial
    {
        /**
         * @brief Throws unless every kernel, stride and dilation is positive and no padding is negative.
         * @throws std::invalid_argument naming the first field that breaks the rule.
         * @see modules/operations/spatial/README.md
         */
        inline void require_valid_geometry(const ConvGeometry& g, const char* name)
        {
            const struct
            {
                const char* field;
                int64_t value;
                int64_t minimum;
            } rules[] = {
                {"kernel_h", g.kernel_h, 1},     {"kernel_w", g.kernel_w, 1},
                {"stride_h", g.stride_h, 1},     {"stride_w", g.stride_w, 1},
                {"dilation_h", g.dilation_h, 1}, {"dilation_w", g.dilation_w, 1},
                {"pad_top", g.pad_top, 0},       {"pad_left", g.pad_left, 0},
            };

            for (const auto& rule : rules)
            {
                if (rule.value < rule.minimum)
                    Dispatch::Detail::throw_operand_error(name, "geometry",
                                                          "has {} = {}; it must be at least {}",
                                                          rule.field, rule.value, rule.minimum);
            }
        }

        /**
         * @brief Returns true when a window whose first tap sits at origin has a tap inside [0, in).
         * @pre origin < in, dilation > 0, kernel > 0.
         */
        [[nodiscard]] constexpr bool window_has_live_tap(int64_t origin, int64_t dilation,
                                                         int64_t kernel, int64_t in) noexcept
        {
            if (origin >= 0)
                return true;

            const int64_t first_live = (-origin + dilation - 1) / dilation;
            return first_live < kernel && origin + first_live * dilation < in;
        }

        /**
         * @brief Throws unless every window along one axis has at least one in-bounds tap.
         * @throws std::invalid_argument naming the first window made only of padding.
         * @see modules/operations/spatial/README.md
         */
        inline void require_live_windows(int64_t in, int64_t out, int64_t pad, int64_t dilation,
                                         int64_t kernel, int64_t stride, const char* axis,
                                         const char* name)
        {
            // Only windows whose origin is negative can miss the input; the rest start inside it.
            for (int64_t o = 0; o < out && o * stride - pad < 0; ++o)
            {
                if (!window_has_live_tap(o * stride - pad, dilation, kernel, in))
                    Dispatch::Detail::throw_operand_error(
                        name, "geometry",
                        "puts output {} {} entirely in the padding (padding {}, kernel {}, "
                        "dilation {}, input extent {}); a pooling window needs a real tap",
                        axis, o, pad, kernel, dilation, in);
            }
        }

        /**
         * @brief Describes a windowed operation whose output extents the caller states.
         * @return The span in element counts; the `out_*_stride` fields stay zero.
         * @throws std::invalid_argument on a bad geometry, an output extent below 1, an in that is
         *         not rank 4, or a window origin outside the padded input.
         * @see modules/operations/spatial/README.md
         */
        [[nodiscard]] inline WindowSpan window_span_for(const Tensor& in, const ConvGeometry& g,
                                                        int64_t out_h, int64_t out_w,
                                                        const char* name)
        {
            require_valid_geometry(g, name);

            if (out_h < 1 || out_w < 1)
                Dispatch::Detail::throw_operand_error(name, "output",
                                                      "has extents {}x{}; both must be at least 1",
                                                      out_h, out_w);

            Dispatch::require_rank(in, 4, name, "in");

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

            // A window that starts past the last input row is refused; see modules/operations/spatial/README.md.
            Dispatch::require_operand(
                !(out_h > 0 && (out_h - 1) * g.stride_h - g.pad_top >= span.in_h), name, "in",
                "has {} rows padded by {} above, but output row {} would start at input row {}",
                span.in_h, g.pad_top, out_h - 1, (out_h - 1) * g.stride_h - g.pad_top);
            Dispatch::require_operand(
                !(out_w > 0 && (out_w - 1) * g.stride_w - g.pad_left >= span.in_w), name, "in",
                "has {} columns padded by {} on the left, but output column {} would start at "
                "input column {}",
                span.in_w, g.pad_left, out_w - 1, (out_w - 1) * g.stride_w - g.pad_left);

            return span;
        }

        /**
         * @brief Describes a windowed operation from its two rank-4 NHWC operands.
         * @return The span in element counts, with both operands' strides filled in.
         * @throws std::invalid_argument on a rank other than 4, a batch or channel mismatch, or a
         *         bad origin.
         * @see modules/operations/spatial/README.md
         */
        [[nodiscard]] inline WindowSpan window_span_for(const Tensor& in, const ConvGeometry& g,
                                                        const Tensor& out, const char* name)
        {
            Dispatch::require_rank(out, 4, name, "out");

            WindowSpan span = window_span_for(in, g, out.shape[1], out.shape[2], name);

            Dispatch::require_extent(out.shape[0], span.batch, "images", name, "out");
            Dispatch::require_extent(out.shape[3], span.channels, "channels", name, "out");

            span.out_batch_stride = out.strides[0];
            span.out_row_stride = out.strides[1];
            span.out_col_stride = out.strides[2];
            span.out_channel_stride = out.strides[3];

            return span;
        }

        /**
         * @brief Describes a pooling operation from its two operands and refuses all-padding windows.
         * @return The span window_span_for returns.
         * @throws std::invalid_argument when some window holds only padding, or as window_span_for.
         * @see modules/operations/spatial/README.md
         */
        [[nodiscard]] inline WindowSpan pooling_span_for(const Tensor& in, const ConvGeometry& g,
                                                         const Tensor& out, const char* name)
        {
            const WindowSpan span = window_span_for(in, g, out, name);

            require_live_windows(span.in_h, span.out_h, g.pad_top, g.dilation_h, g.kernel_h,
                                 g.stride_h, "row", name);
            require_live_windows(span.in_w, span.out_w, g.pad_left, g.dilation_w, g.kernel_w,
                                 g.stride_w, "column", name);

            return span;
        }

        /**
         * @brief Describes an im2col matrix's addressing and checks its extents.
         * @throws std::invalid_argument when cols is not rank 2 or not `span.windows()` by
         *         `span.patch_elements()`.
         */
        [[nodiscard]] inline ColumnSpan column_span_for(const Tensor& cols, const WindowSpan& span,
                                                        const char* name)
        {
            Dispatch::require_rank(cols, 2, name, "cols");
            Dispatch::require_extent(cols.shape[0], span.windows(), "rows", name, "cols");
            Dispatch::require_extent(cols.shape[1], span.patch_elements(), "columns", name, "cols");

            return ColumnSpan{cols.strides[0], cols.strides[1]};
        }

        /**
         * @brief Describes a per-channel tap tensor's addressing and checks its extents.
         * @param operand The tensor's name in the caller's signature, for the message.
         * @throws std::invalid_argument when taps is not rank 3 or not `kernel_h` by `kernel_w` by
         *         `span.channels`.
         * @see modules/operations/spatial/README.md
         */
        [[nodiscard]] inline TapSpan tap_span_for(const Tensor& taps, const WindowSpan& span,
                                                  const char* name, const char* operand)
        {
            Dispatch::require_rank(taps, 3, name, operand);
            Dispatch::require_extent(taps.shape[0], span.geometry.kernel_h, "rows", name, operand);
            Dispatch::require_extent(taps.shape[1], span.geometry.kernel_w, "columns", name,
                                     operand);
            Dispatch::require_extent(taps.shape[2], span.channels, "channels", name, operand);

            return TapSpan{taps.strides[0], taps.strides[1], taps.strides[2]};
        }

    } // namespace Spatial
} // namespace SushiBLAS
