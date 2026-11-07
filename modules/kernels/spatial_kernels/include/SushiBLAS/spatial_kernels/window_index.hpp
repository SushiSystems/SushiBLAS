/**************************************************************************/
/* window_index.hpp                                                       */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file window_index.hpp
 * @brief Declares the sliding-window index math of the spatial kernels, shared by the host and the device.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/pixel_index.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Indexing
        {
            /** @brief Returns the input coordinate a tap reads for output position 0: `tap * dilation - pad_before`. */
            [[nodiscard]] constexpr int64_t window_origin(int64_t tap, int64_t dilation,
                                                          int64_t pad_before) noexcept
            {
                return tap * dilation - pad_before;
            }

            /** @brief Returns the input coordinate a tap reads for an output position, given the tap's origin. */
            [[nodiscard]] constexpr int64_t origin_coord(int64_t out_pos, int64_t stride,
                                                         int64_t origin) noexcept
            {
                return out_pos * stride + origin;
            }

            /**
             * @brief Returns the input coordinate along one axis that tap `tap` of the window at
             *        output position `out_pos` reads; it may fall in the padding.
             * @see kernels/README.md
             */
            [[nodiscard]] constexpr int64_t gather_coord(int64_t out_pos, int64_t stride,
                                                         int64_t pad_before, int64_t tap,
                                                         int64_t dilation) noexcept
            {
                return origin_coord(out_pos, stride, window_origin(tap, dilation, pad_before));
            }

            /** @brief Returns whether a coordinate lies inside `[0, extent)`; outside is padding. */
            [[nodiscard]] constexpr bool in_extent(int64_t pos, int64_t extent) noexcept
            {
                return pos >= 0 && pos < extent;
            }

            /**
             * @brief Returns the output position along one axis whose window reads input position
             *        `in_pos` through tap `tap`, or -1 when no window does.
             * @param stride Must be positive.
             * @see kernels/README.md
             */
            [[nodiscard]] constexpr int64_t scatter_coord(int64_t in_pos, int64_t stride,
                                                          int64_t pad_before, int64_t tap,
                                                          int64_t dilation,
                                                          int64_t out_extent) noexcept
            {
                const int64_t numerator = in_pos + pad_before - tap * dilation;
                if (numerator < 0 || numerator % stride != 0)
                    return -1;
                const int64_t out_pos = numerator / stride;
                return out_pos >= out_extent ? -1 : out_pos;
            }

            /** @brief Returns how many taps one window has: `kernel_h * kernel_w`. */
            [[nodiscard]] constexpr int64_t window_area(const Spatial::ConvGeometry& g) noexcept
            {
                return g.kernel_h * g.kernel_w;
            }

            /** @brief Holds one window's position: image, output row and output column. */
            struct WindowCoord
            {
                int64_t n = 0;      /**< Image index. */
                int64_t h_o = 0;    /**< Output row. */
                int64_t w_o = 0;    /**< Output column. */
            };

            /** @brief Splits a flat window index, column fastest, into `(n, h_o, w_o)`. */
            [[nodiscard]] constexpr WindowCoord window_coord(int64_t flat, int64_t out_h,
                                                             int64_t out_w) noexcept
            {
                const PixelCoord at = pixel_coord(flat, out_h, out_w, 1);
                return WindowCoord{at.n, at.h, at.w};
            }

            /** @brief Returns the flat window index, the inverse of window_coord: `(n * out_h + h_o) * out_w + w_o`. */
            [[nodiscard]] constexpr int64_t window_row(const WindowCoord& at, int64_t out_h,
                                                       int64_t out_w) noexcept
            {
                return (at.n * out_h + at.h_o) * out_w + at.w_o;
            }

            /** @brief Holds one patch element's position: tap row, tap column and channel. */
            struct TapCoord
            {
                int64_t r = 0;  /**< Tap row. */
                int64_t s = 0;  /**< Tap column. */
                int64_t c = 0;  /**< Channel. */
            };

            /** @brief Splits a flat patch index, channel fastest, into `(r, s, c)`. */
            [[nodiscard]] constexpr TapCoord tap_coord(int64_t flat, int64_t channels,
                                                       int64_t kernel_w) noexcept
            {
                const int64_t rs = flat / channels;
                return TapCoord{rs / kernel_w, rs % kernel_w, flat % channels};
            }

            /** @brief Returns the flat patch index, the inverse of tap_coord: `(r * kernel_w + s) * channels + c`. */
            [[nodiscard]] constexpr int64_t patch_column(const TapCoord& tap, int64_t channels,
                                                         int64_t kernel_w) noexcept
            {
                return (tap.r * kernel_w + tap.s) * channels + tap.c;
            }

            /** @brief Returns the offset of one entry of an im2col matrix. */
            [[nodiscard]] constexpr int64_t column_offset(const Spatial::ColumnSpan& cols,
                                                          int64_t row, int64_t col) noexcept
            {
                return row * cols.row_stride + col * cols.element_stride;
            }

            /** @brief Returns the offset of tap `(r, s, c)` in a per-channel tap tensor. */
            [[nodiscard]] constexpr int64_t tap_offset(const Spatial::TapSpan& taps,
                                                       const TapCoord& tap) noexcept
            {
                return tap.r * taps.row_stride + tap.s * taps.col_stride
                     + tap.c * taps.channel_stride;
            }
        }
    }
}
