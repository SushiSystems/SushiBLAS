/**************************************************************************/
/* csr_pattern.hpp                                                        */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file csr_pattern.hpp
 * @brief Declares CsrPattern, how a kernel reads the rows and columns of a CSR matrix.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Holds the shape, the index base and the two index arrays of one CSR matrix.
         * @tparam Index int32_t or int64_t, the element type of both arrays.
         * @see kernels/README.md
         */
        template<typename Index>
        struct CsrPattern
        {
            int64_t rows = 0;                      /**< Number of rows. */
            int64_t cols = 0;                      /**< Number of columns. */
            int64_t nnz = 0;                       /**< Number of stored entries. */
            int64_t first_index = 0;               /**< The stored value of row 0 and column 0. */
            const Index* row_offsets = nullptr;    /**< rows + 1 offsets into the entry arrays. */
            const Index* column_indices = nullptr; /**< The column of each stored entry. */

            /**
             * @brief Returns the zero-based position of the first entry of a row.
             * @pre row lies in [0, rows].
             */
            [[nodiscard]] constexpr int64_t row_begin(int64_t row) const noexcept
            {
                return static_cast<int64_t>(row_offsets[row]) - first_index;
            }

            /**
             * @brief Returns the zero-based position one past the last entry of a row.
             * @pre row lies in [0, rows).
             */
            [[nodiscard]] constexpr int64_t row_end(int64_t row) const noexcept
            {
                return static_cast<int64_t>(row_offsets[row + 1]) - first_index;
            }

            /**
             * @brief Returns the zero-based column of a stored entry.
             * @pre entry lies in [0, nnz).
             */
            [[nodiscard]] constexpr int64_t column(int64_t entry) const noexcept
            {
                return static_cast<int64_t>(column_indices[entry]) - first_index;
            }
        };
    } // namespace Kernels
} // namespace SushiBLAS
