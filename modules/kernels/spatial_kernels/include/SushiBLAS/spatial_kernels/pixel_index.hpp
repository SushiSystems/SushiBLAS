/**************************************************************************/
/* pixel_index.hpp                                                        */
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
 * @file pixel_index.hpp
 * @brief Declares the flat-index and offset math of an NHWC tensor, shared by the host and the device.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/geometry/spatial_geometry.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Indexing
        {
            /** @brief Holds one element's logical position in an `[N, H, W, C]` tensor. */
            struct PixelCoord
            {
                int64_t n = 0;  /**< Image index. */
                int64_t h = 0;  /**< Row. */
                int64_t w = 0;  /**< Column. */
                int64_t c = 0;  /**< Channel. */
            };

            /**
             * @brief Splits a flat element index, channel fastest, into its `(n, h, w, c)` position.
             * @param flat Index in `[0, N * height * width * channels)`.
             * @param height Must be positive.
             * @param width Must be positive.
             * @param channels Must be positive.
             * @see kernels/README.md
             */
            [[nodiscard]] constexpr PixelCoord pixel_coord(int64_t flat, int64_t height,
                                                           int64_t width, int64_t channels) noexcept
            {
                const int64_t c = flat % channels;
                const int64_t hw = flat / channels;
                const int64_t w = hw % width;
                const int64_t nh = hw / width;
                return PixelCoord{nh / height, nh % height, w, c};
            }

            /**
             * @brief Returns the dense row-major flat index of a position: `((n * height + h) * width + w) * channels + c`.
             * @see kernels/README.md
             */
            [[nodiscard]] constexpr int64_t flat_pixel(const PixelCoord& at, int64_t height,
                                                       int64_t width, int64_t channels) noexcept
            {
                return ((at.n * height + at.h) * width + at.w) * channels + at.c;
            }

            /** @brief Holds the four element strides of one tensor; it is trivially copyable into a kernel. */
            struct Strides
            {
                int64_t batch = 0;      /**< Elements between consecutive images. */
                int64_t row = 0;        /**< Elements between vertically adjacent pixels. */
                int64_t col = 0;        /**< Elements between horizontally adjacent pixels. */
                int64_t channel = 1;    /**< Elements between adjacent channels of one pixel. */

                /** @brief Returns the offset of `(n, h, w, c)`. */
                [[nodiscard]] constexpr int64_t offset(int64_t n, int64_t h, int64_t w,
                                                       int64_t c) const noexcept
                {
                    return n * batch + h * row + w * col + c * channel;
                }
            };

            /** @brief Returns the strides of the span's input. */
            [[nodiscard]] constexpr Strides input_strides(const Spatial::WindowSpan& span) noexcept
            {
                return Strides{span.in_batch_stride, span.in_row_stride, span.in_col_stride,
                               span.in_channel_stride};
            }

            /** @brief Returns the strides of the span's output. */
            [[nodiscard]] constexpr Strides output_strides(const Spatial::WindowSpan& span) noexcept
            {
                return Strides{span.out_batch_stride, span.out_row_stride, span.out_col_stride,
                               span.out_channel_stride};
            }

            /** @brief Returns the offset of element `(n, c)` of an `[N, C]` tensor. */
            [[nodiscard]] constexpr int64_t batch_channel_offset(int64_t n, int64_t c,
                                                                 int64_t batch_stride,
                                                                 int64_t channel_stride) noexcept
            {
                return n * batch_stride + c * channel_stride;
            }
        }
    }
}
