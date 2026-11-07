/**************************************************************************/
/* blend_index.hpp                                                        */
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
 * @file blend_index.hpp
 * @brief Declares the index math and the homogeneous-point step of the transform blend, shared by the host and the device.
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
            /** @brief Returns the offset of point `p` in a buffer of `components` values per point. */
            [[nodiscard]] constexpr int64_t point_offset(int64_t p, int64_t components) noexcept
            {
                return p * components;
            }

            /** @brief Returns the offset of influence `j` of point `p`: `p * influences + j`. */
            [[nodiscard]] constexpr int64_t influence_offset(int64_t p, int64_t influences,
                                                             int64_t j) noexcept
            {
                return p * influences + j;
            }

            /** @brief Returns whether a transform slot is inside `[0, transform_count)`; outside adds no term. */
            [[nodiscard]] constexpr bool slot_live(int64_t slot, int64_t transform_count) noexcept
            {
                return slot >= 0 && slot < transform_count;
            }

            /** @brief Returns the offset of transform `slot` in a `[T][rows][4]` buffer. */
            [[nodiscard]] constexpr int64_t matrix_offset(int64_t slot,
                                                          int64_t transform_rows) noexcept
            {
                return slot * transform_rows * 4;
            }

            /** @brief Returns the offset of matrix element `(row, col)` within one transform. */
            [[nodiscard]] constexpr int64_t matrix_element(int64_t row, int64_t col) noexcept
            {
                return row * 4 + col;
            }

            /** @brief Returns whether output component `i` comes from a stored matrix row; otherwise it is the implied last row `(0, 0, 0, 1)`. */
            [[nodiscard]] constexpr bool has_matrix_row(int64_t i, int64_t transform_rows) noexcept
            {
                return i < transform_rows;
            }

            /**
             * @brief Returns the w of a point: its fourth value when homogeneous, else 1.
             * @tparam W The type the point is formed in.
             * @param components 3 (a position) or 4 (homogeneous).
             * @see kernels/README.md
             */
            template<typename W, typename T>
            [[nodiscard]] inline W homogeneous_w(const T* point, int64_t components)
            {
                return components == 4 ? W(point[3]) : W(1);
            }
        }
    }
}
