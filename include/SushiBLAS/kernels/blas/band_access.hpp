/**************************************************************************/
/* band_access.hpp                                                        */
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
 * @file band_access.hpp
 * @brief Declares BandAccess, how a kernel reads element (row, col) of a band-stored op(X).
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Holds the steps, the origin, the band widths and the conjugation of a band op(X).
         * @see include/SushiBLAS/kernels/README.md
         */
        struct BandAccess
        {
            int64_t row = 0;         /**< Elements to advance to move down one row of op(X). */
            int64_t col = 1;         /**< Elements to advance to move right one column of op(X). */
            int64_t origin = 0;      /**< Offset of element (0, 0) of op(X) in the storage. */
            int64_t lower = 0;       /**< Sub-diagonals of op(X) the storage holds. */
            int64_t upper = 0;       /**< Super-diagonals of op(X) the storage holds. */
            bool conjugated = false; /**< Whether a complex element is conjugated on read. */

            /** @brief Returns the offset of element (r, c) of op(X) from the first one stored. */
            [[nodiscard]] constexpr int64_t index(int64_t r, int64_t c) const noexcept
            {
                return origin + r * row + c * col;
            }

            /** @brief Returns whether element (r, c) of op(X) lies inside the stored band. */
            [[nodiscard]] constexpr bool holds(int64_t r, int64_t c) const noexcept
            {
                return c - r <= upper && r - c <= lower;
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

            /** @brief Returns the first column of row r of op(X) the storage holds. */
            [[nodiscard]] constexpr int64_t first_column(int64_t r) const noexcept
            {
                return r > lower ? r - lower : 0;
            }

            /**
             * @brief Returns one past the last column of row r of op(X) the storage holds.
             * @param columns Columns of op(X).
             * @return A value at most first_column(r) when the row holds nothing.
             */
            [[nodiscard]] constexpr int64_t end_column(int64_t r, int64_t columns) const noexcept
            {
                return columns - r > upper + 1 ? r + upper + 1 : columns;
            }

            /** @brief Returns the access of the transpose of op(X), with the same conjugation. */
            [[nodiscard]] constexpr BandAccess transposed() const noexcept
            {
                return BandAccess{col, row, origin, upper, lower, conjugated};
            }
        };

        /**
         * @brief Resolves a layout, a Transpose, two band widths and a leading dimension into a
         *        BandAccess.
         * @param kl Sub-diagonals of X, not of op(X); ku counts its super-diagonals.
         * @param ld Elements stored per column of X (column-major) or per row (row-major).
         * @pre kl and ku are not negative, and ld is at least kl + ku + 1.
         * @see include/SushiBLAS/kernels/README.md
         */
        [[nodiscard]] inline constexpr BandAccess band_access(Core::Layout layout,
                                                              Core::Transpose trans, int64_t kl,
                                                              int64_t ku, int64_t ld) noexcept
        {
            const BandAccess stored = (layout == Core::Layout::ColumnMajor)
                                          ? BandAccess{1, ld - 1, ku, kl, ku, false}
                                          : BandAccess{ld - 1, 1, kl, kl, ku, false};
            if (!Core::is_transposed(trans))
                return stored;

            BandAccess op = stored.transposed();
            op.conjugated = Core::is_conjugated(trans);
            return op;
        }

        /**
         * @brief Resolves the band storage of one triangle with k off-diagonals into a BandAccess.
         * @param uplo The triangle of X, not of op(X), that the storage holds.
         * @pre k is not negative, and ld is at least k + 1.
         * @see include/SushiBLAS/kernels/README.md
         */
        [[nodiscard]] inline constexpr BandAccess triangle_band_access(Core::Layout layout,
                                                                       Core::Uplo uplo,
                                                                       Core::Transpose trans,
                                                                       int64_t k,
                                                                       int64_t ld) noexcept
        {
            const bool upper = (uplo == Core::Uplo::Upper);
            return band_access(layout, trans, upper ? int64_t{0} : k, upper ? k : int64_t{0}, ld);
        }
    } // namespace Kernels
} // namespace SushiBLAS
