/**************************************************************************/
/* im2col_index.hpp                                                       */
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
 * @file im2col_index.hpp
 * @brief Declares the index math of one element of an im2col matrix, shared by the host and the device.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/spatial_kernels/window_index.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Indexing
        {
            /** @brief Holds where one im2col element sits in the matrix and in the image. */
            struct Im2colCoord
            {
                int64_t row = 0;        /**< Matrix row: the flat window index. */
                int64_t col = 0;        /**< Matrix column: the flat patch index. */
                WindowCoord window{};   /**< The window the row stands for. */
                TapCoord tap{};         /**< The patch element the column stands for. */
            };

            /**
             * @brief Splits a flat matrix element index, patch fastest, into row, column, window and tap.
             * @param patch_width Matrix columns, `kernel_h * kernel_w * channels`; must be positive.
             * @see kernels/README.md
             */
            [[nodiscard]] constexpr Im2colCoord im2col_coord(int64_t flat, int64_t patch_width,
                                                             int64_t channels, int64_t kernel_w,
                                                             int64_t out_h, int64_t out_w) noexcept
            {
                const int64_t row = flat / patch_width;
                const int64_t col = flat % patch_width;
                return Im2colCoord{row, col, window_coord(row, out_h, out_w),
                                   tap_coord(col, channels, kernel_w)};
            }
        }
    }
}
