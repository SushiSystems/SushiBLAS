/**************************************************************************/
/* rank_update_index.hpp                                                  */
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

/**
 * @file rank_update_index.hpp
 * @brief Declares the layout offset, work-item split and triangle tests the rank updates share, usable on the host and the device.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/blas_seam_kernels/extent_index.hpp>
#include <SushiBLAS/blas_seam_kernels/gemm_index.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Indexing
        {
            /** @brief Returns the offset of element (i, j) in a matrix of leading dimension `ld`. */
            [[nodiscard]] constexpr int64_t layout_offset(bool row_major, int64_t i, int64_t j,
                                                          int64_t ld) noexcept
            {
                return row_major ? row_major_offset(i, ld, j) : row_major_offset(j, ld, i);
            }

            /**
             * @brief Splits a two-dimensional work-item id into `(row, col)`, the second id being
             *        the row when the elements of A are adjacent down the columns.
             */
            [[nodiscard]] constexpr GridCoord element_coord(int64_t first, int64_t second,
                                                            bool by_columns) noexcept
            {
                return by_columns ? GridCoord{second, first} : GridCoord{first, second};
            }

            /** @brief Returns whether element (i, j) lies outside the triangle the update writes. */
            [[nodiscard]] constexpr bool outside_triangle(bool upper, int64_t i, int64_t j) noexcept
            {
                return upper ? (j < i) : (j > i);
            }

            /** @brief Returns whether a rows x cols block at `(row0, col0)` holds part of the triangle. */
            [[nodiscard]] constexpr bool block_touches_triangle(bool upper, int64_t row0,
                                                                int64_t rows, int64_t col0,
                                                                int64_t cols) noexcept
            {
                return upper ? (col0 + cols - 1 >= row0) : (row0 + rows - 1 >= col0);
            }
        }
    }
}
