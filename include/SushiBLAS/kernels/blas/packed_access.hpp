/**************************************************************************/
/* packed_access.hpp                                                      */
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
 * @file packed_access.hpp
 * @brief Declares PackedAccess, how a kernel reads element (row, col) of a packed op(X).
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/kernels/blas/row_cursor.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Holds the order, the packing direction and the conjugation of a packed op(X).
         * @see include/SushiBLAS/kernels/README.md
         */
        struct PackedAccess
        {
            int64_t order = 0;       /**< Rows, and columns, of the matrix. */
            bool by_rows = false;    /**< Whether a stored line is a row of op(X), not a column. */
            bool growing = true;     /**< Whether line p holds p + 1 elements, not order - p. */
            bool conjugated = false; /**< Whether a complex element is conjugated on read. */

            /** @brief Returns whether the stored triangle of op(X) is the upper one. */
            [[nodiscard]] constexpr bool is_upper() const noexcept
            {
                return growing != by_rows;
            }

            /** @brief Returns whether element (r, c) of op(X) lies inside the stored triangle. */
            [[nodiscard]] constexpr bool holds(int64_t r, int64_t c) const noexcept
            {
                return is_upper() ? r <= c : r >= c;
            }

            /**
             * @brief Returns the offset of element (r, c) of op(X) from the first one stored.
             * @pre holds(r, c).
             */
            [[nodiscard]] constexpr int64_t index(int64_t r, int64_t c) const noexcept
            {
                const int64_t line = by_rows ? r : c;
                const int64_t along = by_rows ? c : r;
                return growing ? along + line * (line + 1) / 2
                               : along + line * (2 * order - line - 1) / 2;
            }

            /**
             * @brief Returns element (r, c) of op(X).
             * @pre holds(r, c), and (r, c) lies inside op(X).
             */
            template<typename T>
            [[nodiscard]] constexpr T at(const T* p, int64_t r, int64_t c) const noexcept
            {
                return conjugate_if(conjugated, p[index(r, c)]);
            }

            /**
             * @brief Returns a cursor on element (r, c) of op(X) that walks row r.
             * @pre holds(r, c) before the cursor's offset is read.
             */
            [[nodiscard]] constexpr RowCursor row_cursor(int64_t r, int64_t c) const noexcept
            {
                if (by_rows)
                    return RowCursor{index(r, c), 1, 0};
                return growing ? RowCursor{index(r, c), c + 1, 1}
                               : RowCursor{index(r, c), order - c - 1, -1};
            }

            /**
             * @brief Returns the element of op(X) a cursor stands on.
             * @pre The cursor stands inside the stored triangle.
             */
            template<typename T>
            [[nodiscard]] constexpr T at(const T* p, const RowCursor& cursor) const noexcept
            {
                return conjugate_if(conjugated, p[cursor.offset]);
            }

            /** @brief Returns the first column of row r of op(X) the storage holds. */
            [[nodiscard]] constexpr int64_t first_column(int64_t r) const noexcept
            {
                return is_upper() ? r : 0;
            }

            /**
             * @brief Returns one past the last column of row r of op(X) the storage holds.
             * @param columns Columns of op(X), which is order.
             */
            [[nodiscard]] constexpr int64_t end_column(int64_t r, int64_t columns) const noexcept
            {
                return is_upper() ? columns : r + 1;
            }

            /** @brief Returns the access of the transpose of op(X), with the same conjugation. */
            [[nodiscard]] constexpr PackedAccess transposed() const noexcept
            {
                return PackedAccess{order, !by_rows, growing, conjugated};
            }
        };

        /**
         * @brief Resolves a layout, a Uplo, a Transpose and an order into a PackedAccess.
         * @param uplo The triangle of X, not of op(X), that the storage holds.
         * @pre n is not negative.
         * @see include/SushiBLAS/kernels/README.md
         */
        [[nodiscard]] inline constexpr PackedAccess packed_access(Core::Layout layout,
                                                                  Core::Uplo uplo,
                                                                  Core::Transpose trans,
                                                                  int64_t n) noexcept
        {
            const bool by_rows = (layout == Core::Layout::RowMajor);
            const PackedAccess stored{n, by_rows, (uplo == Core::Uplo::Upper) != by_rows, false};
            if (!Core::is_transposed(trans))
                return stored;

            PackedAccess op = stored.transposed();
            op.conjugated = Core::is_conjugated(trans);
            return op;
        }
    } // namespace Kernels
} // namespace SushiBLAS
