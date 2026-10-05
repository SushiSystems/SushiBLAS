/****************************************************************/
/* geometry.hpp                                                 */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#pragma once

/**
 * @file geometry.hpp
 * @brief The window geometry of a spatial operation, and the extents it implies.
 *
 * Separate from `window.hpp` and free of `Tensor` on purpose. The output-extent
 * formula below is the *only* statement of how an input extent, a padding pair,
 * a dilation, a kernel and a stride combine, and it has two consumers on either
 * side of a repository boundary: this library's kernels, and SushiAI's host-lane
 * shape inference. That inference is device-free by design - it is exercised in
 * a unit tier that links no SYCL - so the arithmetic has to be reachable without
 * dragging the device stack behind it. Two copies of one formula that must agree
 * is the alternative, and they would agree until the day they did not.
 *
 * Includes `<cstdint>` and nothing else. Keep it that way.
 */

#include <cstdint>

namespace SushiBLAS
{
    namespace Spatial
    {
        /**
         * @struct ConvGeometry
         * @brief A 2-D sliding window: its size, its step, its spacing, its origin.
         *
         * Exactly eight numbers, and the count is load-bearing rather than tidy.
         * SushiAI carries this into `Graph::Attributes`, whose capacity is
         * bounded by `SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS` - eight.
         *
         * **`pad_bottom` and `pad_right` are deliberately absent.** A kernel
         * addresses its input as `h_in = h_o * stride_h - pad_top + r * dilation_h`
         * and bounds-checks against `[0, H)`. The trailing pads never enter that
         * expression; they affect only *how many output rows exist*, which is
         * @ref output_extent's answer and which the host has already written into
         * the output tensor's shape by the time any of this reaches a device. So
         * asymmetric padding costs zero slots over symmetric padding, and every
         * geometry is expressible: non-square kernels, anisotropic stride and
         * dilation, and independent padding on all four sides.
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
         * @brief How many windows fit along one axis.
         *
         * `floor((in + pad_before + pad_after - dilation*(kernel - 1) - 1) / stride) + 1`,
         * which is the same expression PyTorch, Darknet and cuDNN all use.
         *
         * A window wider than the padded input answers **0** rather than letting
         * the numerator go negative. That is not defensiveness: the result is
         * multiplied into an allocation size two layers up, and a negative
         * numerator under integer division would produce a plausible-looking
         * positive extent for some operand combinations.
         *
         * @param in The input extent along this axis.
         * @param pad_before Implicit zeros before index 0.
         * @param pad_after Implicit zeros after the last index.
         * @param dilation Spacing between taps; 1 for an undilated window.
         * @param kernel The window's extent along this axis.
         * @param stride The step between consecutive windows.
         * @return The output extent, or 0 when no window fits.
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
         * @struct WindowSpan
         * @brief Where one spatial operation's input and output live, in element counts.
         *
         * Modelled on @ref SushiBLAS::Reduce::AxisSpan, and for the same reason:
         * it carries counts and strides only, so the layout, the axis order and
         * any non-unit stride a view inherited are resolved once on the host and
         * the device sees none of them. One kernel per operation, no layout
         * branch in the inner loop, no second instantiation.
         *
         * The struct names no layout. Every caller in the convolutional stack
         * builds it from an NHWC tensor (see the spec's CV-3), but adding NCHW
         * later is filling in a different stride assignment in
         * `Spatial::window_span_for`, not re-cutting this abstraction.
         *
         * All strides are element counts and all are non-negative, which
         * `Tensor`'s stride-explicit constructor already guarantees.
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
         * @struct ColumnSpan
         * @brief How an im2col matrix is addressed: a row stride and an element stride.
         *
         * Kept out of @ref WindowSpan because the column matrix is rank-2 and
         * everything else in this family is rank-4. Folding two unrelated
         * addressing schemes into one struct would leave four fields meaningless
         * for every operation but two.
         *
         * Entry `(row, element)` lives at `data[row * row_stride + element * element_stride]`,
         * where `row` indexes a window in `(n, h_o, w_o)` order and `element`
         * indexes a patch position in `(r, s, c)` order - channel innermost, so
         * an NHWC gather copies contiguous runs of `channels` elements.
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
