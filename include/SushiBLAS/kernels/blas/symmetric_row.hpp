/**************************************************************************/
/* symmetric_row.hpp                                                      */
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
 * @file symmetric_row.hpp
 * @brief Declares SymmetricRow, which walks one row of a symmetric matrix stored as a triangle.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/kernels/blas/row_cursor.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Walks one row of a symmetric or Hermitian matrix of which one triangle is stored,
         *        in ascending column order.
         * @tparam Conj Unconjugated for a symmetric matrix, Conjugated for a Hermitian one.
         * @tparam Access BandAccess or PackedAccess of the stored triangle.
         * @see include/SushiBLAS/kernels/README.md
         */
        template<typename Conj, typename Access>
        struct SymmetricRow
        {
            Access stored;       /**< How the stored triangle is read. */
            Access mirror;       /**< How the transpose of the stored triangle is read. */
            int64_t columns = 0; /**< Rows, and columns, of the matrix. */

            /** @brief Returns whether the stored triangle holds the columns left of (i, i). */
            [[nodiscard]] constexpr bool stores_below(int64_t i) const noexcept
            {
                return stored.first_column(i) < i;
            }

            /** @brief Returns whether the stored triangle holds the columns right of (i, i). */
            [[nodiscard]] constexpr bool stores_above(int64_t i) const noexcept
            {
                return stored.end_column(i, columns) > i + 1;
            }

            /** @brief Returns the first column of row i that is not zero by structure. */
            [[nodiscard]] constexpr int64_t first_column(int64_t i) const noexcept
            {
                return (stores_below(i) ? stored : mirror).first_column(i);
            }

            /** @brief Returns one past the last column of row i that is not zero by structure. */
            [[nodiscard]] constexpr int64_t end_column(int64_t i) const noexcept
            {
                return (stores_above(i) ? stored : mirror).end_column(i, columns);
            }

            /**
             * @brief Calls visit(value, column) for every element of row i from first_column(i)
             *        to end_column(i).
             * @param p The stored triangle.
             */
            template<typename T, typename Visit>
            constexpr void for_each(const T* p, int64_t i, Visit visit) const
            {
                const bool below_stored = stores_below(i);
                const Access& below = below_stored ? stored : mirror;
                int64_t j = below.first_column(i);
                RowCursor cursor = below.row_cursor(i, j);
                for (; j < i; ++j, cursor.advance())
                {
                    const T value = below.at(p, cursor);
                    visit(below_stored ? value : Conj::apply(value), j);
                }

                visit(Conj::diagonal(stored.at(p, i, i)), i);

                const bool above_stored = stores_above(i);
                const Access& above = above_stored ? stored : mirror;
                const int64_t end = above.end_column(i, columns);
                cursor = above.row_cursor(i, i + 1);
                for (j = i + 1; j < end; ++j, cursor.advance())
                {
                    const T value = above.at(p, cursor);
                    visit(above_stored ? value : Conj::apply(value), j);
                }
            }
        };

        /**
         * @brief Builds the SymmetricRow of a stored triangle of a matrix of order n.
         * @tparam Conj Unconjugated for a symmetric matrix, Conjugated for a Hermitian one.
         * @see include/SushiBLAS/kernels/README.md
         */
        template<typename Conj, typename Access>
        [[nodiscard]] constexpr SymmetricRow<Conj, Access> symmetric_row(const Access& stored,
                                                                         int64_t n) noexcept
        {
            return SymmetricRow<Conj, Access>{stored, stored.transposed(), n};
        }
    } // namespace Kernels
} // namespace SushiBLAS
