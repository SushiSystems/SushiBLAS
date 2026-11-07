/**************************************************************************/
/* packed_product_index.hpp                                               */
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
 * @file packed_product_index.hpp
 * @brief Declares the macro-block and panel index math of the packed product, shared by the host and the device.
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
            /** @brief Holds one macro-block of a packed product: its batch entry and its first row and column. */
            struct PackedBlock
            {
                int64_t batch = 0;  /**< Batch entry. */
                int64_t row0 = 0;   /**< First row of C. */
                int64_t col0 = 0;   /**< First column of C. */
            };

            /**
             * @brief Splits a flat macro-block id, rows fastest inside a matrix, into its batch
             *        entry and first row and column.
             */
            [[nodiscard]] constexpr PackedBlock packed_block(int64_t block, int64_t row_blocks,
                                                             int64_t col_blocks, int64_t block_rows,
                                                             int64_t block_cols) noexcept
            {
                const int64_t per_matrix = row_blocks * col_blocks;
                const int64_t in_matrix = block % per_matrix;
                return PackedBlock{block / per_matrix, (in_matrix % row_blocks) * block_rows,
                                   (in_matrix / row_blocks) * block_cols};
            }

            /** @brief Returns the micro-tile index of register-block group `(g, h)`, row groups fastest. */
            [[nodiscard]] constexpr int64_t micro_tile_index(int64_t g, int64_t h,
                                                             int64_t row_groups) noexcept
            {
                return h * row_groups + g;
            }

            /** @brief Returns the offset in reals of micro-panel `group` in a panel whose micro-panels hold micro_reals each. */
            [[nodiscard]] constexpr int64_t micro_panel_offset(int64_t group,
                                                               int64_t micro_reals) noexcept
            {
                return group * micro_reals;
            }

            /** @brief Returns the offset in reals of the slice of term `term` in a micro-panel. */
            [[nodiscard]] constexpr int64_t slice_offset(int64_t term, int64_t slice_reals) noexcept
            {
                return term * slice_reals;
            }
        }
    }
}
