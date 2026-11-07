/**************************************************************************/
/* transpose_index.hpp                                                    */
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
 * @file transpose_index.hpp
 * @brief Declares the tile geometry and index math of the transposing layout copy, shared by the host and the device.
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
            /** @brief Holds the side of one square tile of the transposing copy. */
            inline constexpr int K_TRANSPOSE_TILE = 32;

            /** @brief Holds the rows of work-items of one transposing work-group. */
            inline constexpr int K_TRANSPOSE_ROWS = 8;

            /**
             * @brief Holds one image as a matrix whose fast axis is adjacent in the input and
             *        whose slow axis is adjacent in the output.
             */
            struct TransposePlan
            {
                int64_t fast = 0;       /**< Extent of the axis adjacent in the input. */
                int64_t slow = 0;       /**< Extent of the axis adjacent in the output. */
                int64_t in_slow = 0;    /**< Input step along the slow axis. */
                int64_t out_fast = 0;   /**< Output step along the fast axis. */
                int64_t in_batch = 0;   /**< Input step from one image to the next. */
                int64_t out_batch = 0;  /**< Output step from one image to the next. */
            };

            /**
             * @brief Returns whether a conversion moves the adjacent axis between pixels and
             *        channels over linear image planes, and fills plan when it does.
             * @see kernels/README.md
             */
            [[nodiscard]] inline bool transposes(const Spatial::WindowSpan& span,
                                                 TransposePlan& plan) noexcept
            {
                const bool planes = span.in_row_stride == span.in_w * span.in_col_stride &&
                                    span.out_row_stride == span.in_w * span.out_col_stride;
                const int64_t pixels = span.in_h * span.in_w;
                plan.in_batch = span.in_batch_stride;
                plan.out_batch = span.out_batch_stride;
                if (planes && span.in_channel_stride == 1 && span.out_col_stride == 1)
                {
                    plan.fast = span.channels;
                    plan.slow = pixels;
                    plan.in_slow = span.in_col_stride;
                    plan.out_fast = span.out_channel_stride;
                    return true;
                }
                if (planes && span.in_col_stride == 1 && span.out_channel_stride == 1)
                {
                    plan.fast = pixels;
                    plan.slow = span.channels;
                    plan.in_slow = span.in_channel_stride;
                    plan.out_fast = span.out_col_stride;
                    return true;
                }
                return false;
            }

            /** @brief Returns how many tiles cover an axis of the given extent; the last may be partial. */
            [[nodiscard]] constexpr int64_t tile_count(int64_t extent) noexcept
            {
                return (extent + K_TRANSPOSE_TILE - 1) / K_TRANSPOSE_TILE;
            }

            /** @brief Returns how many elements one local tile holds, with its padding column. */
            [[nodiscard]] constexpr int64_t tile_elements() noexcept
            {
                return static_cast<int64_t>(K_TRANSPOSE_TILE) * (K_TRANSPOSE_TILE + 1);
            }

            /** @brief Returns the slot of tile row `row`, tile column `col` in the padded local tile. */
            [[nodiscard]] constexpr int64_t tile_slot(int64_t row, int64_t col) noexcept
            {
                return row * (K_TRANSPOSE_TILE + 1) + col;
            }

            /** @brief Returns whether `(slow, fast)` lies inside the image; a lane outside is dead. */
            [[nodiscard]] constexpr bool tile_live(const TransposePlan& plan, int64_t slow,
                                                   int64_t fast) noexcept
            {
                return slow < plan.slow && fast < plan.fast;
            }

            /** @brief Returns the input offset, within one image, of `(slow, fast)`. */
            [[nodiscard]] constexpr int64_t transpose_read_offset(const TransposePlan& plan,
                                                                  int64_t slow,
                                                                  int64_t fast) noexcept
            {
                return slow * plan.in_slow + fast;
            }

            /** @brief Returns the output offset, within one image, of `(slow, fast)`. */
            [[nodiscard]] constexpr int64_t transpose_write_offset(const TransposePlan& plan,
                                                                   int64_t slow,
                                                                   int64_t fast) noexcept
            {
                return slow + fast * plan.out_fast;
            }
        }
    }
}
