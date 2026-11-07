/**************************************************************************/
/* matrix_vector_index.hpp                                                */
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
 * @file matrix_vector_index.hpp
 * @brief Declares the extent, row-shape and row-window math the matrix-vector kernels share, usable on the host and the device.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Indexing
        {
            /** @brief Holds the output and input lengths of a matrix-vector product. */
            struct VectorExtents
            {
                int64_t out_len = 0;    /**< Elements of the result: the rows of op(A). */
                int64_t in_len = 0;     /**< Elements of the vector multiplied: the columns of op(A). */
            };

            /** @brief Returns the lengths of y and x for an m x n matrix A stored as given and op(A) transposed or not. */
            [[nodiscard]] constexpr VectorExtents vector_extents(bool transposed, int64_t m,
                                                                 int64_t n) noexcept
            {
                return transposed ? VectorExtents{n, m} : VectorExtents{m, n};
            }

            /** @brief Returns whether a row of op(A) is contiguous in storage. */
            [[nodiscard]] constexpr bool rows_are_contiguous(bool row_major, bool transposed) noexcept
            {
                return row_major != transposed;
            }

            /** @brief Returns whether fewer rows than one work-group per compute unit leave units idle. */
            [[nodiscard]] constexpr bool rows_underfill_device(int64_t rows, std::size_t units,
                                                               std::size_t group_size) noexcept
            {
                return rows < static_cast<int64_t>(units * group_size);
            }

            /** @brief Returns the most elements a band row holds: the band width, at most `extent`. */
            [[nodiscard]] constexpr int64_t band_row_length(int64_t extent, int64_t lower,
                                                            int64_t upper) noexcept
            {
                const int64_t width = lower + upper + 1;
                return extent < width ? extent : width;
            }

            /** @brief Holds the columns one row folds: first up to, and not including, last. */
            struct RowWindow
            {
                int64_t first = 0;  /**< The first column folded. */
                int64_t last = 0;   /**< One past the last column folded. */
            };

            /** @brief Gives every row the same window, the whole row. */
            struct WholeRow
            {
                int64_t length = 0; /**< Columns in a row. */

                /** @brief Returns the window of a row: every column. */
                [[nodiscard]] constexpr RowWindow operator()(int64_t) const noexcept
                {
                    return RowWindow{0, length};
                }
            };

            /** @brief Gives row i of a triangular matrix its stored columns, diagonal included. */
            struct TriangleRow
            {
                int64_t order = 0;  /**< Rows and columns of the matrix. */
                bool upper = false; /**< Whether row i holds the columns from i on. */

                /** @brief Returns [row, order) for an upper row, [0, row] for a lower one. */
                [[nodiscard]] constexpr RowWindow operator()(int64_t row) const noexcept
                {
                    return upper ? RowWindow{row, order} : RowWindow{0, row + 1};
                }
            };
        }
    }
}
