/**************************************************************************/
/* extent_index.hpp                                                       */
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
 * @file extent_index.hpp
 * @brief Declares the extent, tail and tile-offset math the kernels of this module share, usable on the host and the device.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Indexing
        {
            /**
             * @brief Returns how many blocks of size `block` cover `extent`.
             * @pre extent is not negative and block is positive.
             */
            [[nodiscard]] constexpr int64_t ceil_div(int64_t extent, int64_t block) noexcept
            {
                return (extent + block - 1) / block;
            }

            /**
             * @brief Returns how many elements of a block of `width` starting at `first` lie inside
             *        `total`: the tail guard of a tiled loop.
             * @pre first is not past total.
             */
            [[nodiscard]] constexpr int64_t live_count(int64_t total, int64_t first,
                                                       int64_t width) noexcept
            {
                const int64_t left = total - first;
                return left < width ? left : width;
            }

            /** @brief Returns `row * stride + col`: the offset of an element of a row-major tile. */
            [[nodiscard]] constexpr int64_t row_major_offset(int64_t row, int64_t stride,
                                                             int64_t col) noexcept
            {
                return row * stride + col;
            }
        }
    }
}
