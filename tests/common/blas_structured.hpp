/**************************************************************************/
/* blas_structured.hpp                                                    */
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
 * @file blas_structured.hpp
 * @brief Declares the conversions from a dense test matrix to CBLAS band and packed storage.
 * @author Mustafa Garip
 * @see tests/common/README.md
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Returns a dense m x n matrix, given row by row, in the element order of a layout.
         * @param dense m * n elements, element (i, j) at i * n + j.
         */
        template<typename T>
        [[nodiscard]] std::vector<T> in_layout(const std::vector<T>& dense, int64_t m, int64_t n,
                                               Core::Layout layout)
        {
            if (layout == Core::Layout::RowMajor)
                return dense;

            std::vector<T> out(dense.size());
            for (int64_t i = 0; i < m; ++i)
            {
                for (int64_t j = 0; j < n; ++j)
                    out[static_cast<std::size_t>(i + j * m)] =
                        dense[static_cast<std::size_t>(i * n + j)];
            }
            return out;
        }

        /**
         * @brief Returns the CBLAS band storage of a dense m x n matrix given row by row.
         * @param ld Elements per stored line, at least kl + ku + 1.
         * @param poison Written to every slot that holds no element of the band.
         * @return n lines of ld elements for COLUMN_MAJOR, m lines of ld elements for ROW_MAJOR.
         */
        template<typename T>
        [[nodiscard]] std::vector<T> to_band(const std::vector<T>& dense, int64_t m, int64_t n,
                                             int64_t kl, int64_t ku, int64_t ld,
                                             Core::Layout layout, T poison)
        {
            const bool by_columns = (layout == Core::Layout::ColumnMajor);
            std::vector<T> band(static_cast<std::size_t>((by_columns ? n : m) * ld), poison);

            for (int64_t i = 0; i < m; ++i)
            {
                for (int64_t j = 0; j < n; ++j)
                {
                    if (j - i > ku || i - j > kl)
                        continue;

                    const int64_t offset = by_columns ? j * ld + (ku + i - j)
                                                      : i * ld + (kl + j - i);
                    band[static_cast<std::size_t>(offset)] =
                        dense[static_cast<std::size_t>(i * n + j)];
                }
            }
            return band;
        }

        /**
         * @brief Returns the CBLAS band storage of one triangle of a dense n x n matrix.
         * @param ld Elements per stored line, at least k + 1.
         * @param poison Written to every slot that holds no element of the triangle's band.
         */
        template<typename T>
        [[nodiscard]] std::vector<T> to_triangle_band(const std::vector<T>& dense, int64_t n,
                                                      Core::Uplo uplo, int64_t k, int64_t ld,
                                                      Core::Layout layout, T poison)
        {
            const bool upper = (uplo == Core::Uplo::Upper);
            return to_band(dense, n, n, upper ? int64_t{0} : k, upper ? k : int64_t{0}, ld,
                           layout, poison);
        }

        /**
         * @brief Returns the CBLAS packed storage of one triangle of a dense n x n matrix.
         * @param dense n * n elements, element (i, j) at i * n + j.
         * @return n * (n + 1) / 2 elements: columns one after another for COLUMN_MAJOR, rows for
         *         ROW_MAJOR.
         */
        template<typename T>
        [[nodiscard]] std::vector<T> to_packed(const std::vector<T>& dense, int64_t n,
                                               Core::Uplo uplo, Core::Layout layout)
        {
            const bool upper = (uplo == Core::Uplo::Upper);
            const bool by_columns = (layout == Core::Layout::ColumnMajor);

            std::vector<T> packed;
            packed.reserve(static_cast<std::size_t>(n * (n + 1) / 2));
            for (int64_t line = 0; line < n; ++line)
            {
                for (int64_t along = 0; along < n; ++along)
                {
                    const int64_t i = by_columns ? along : line;
                    const int64_t j = by_columns ? line : along;
                    if (upper ? (i <= j) : (i >= j))
                        packed.push_back(dense[static_cast<std::size_t>(i * n + j)]);
                }
            }
            return packed;
        }
    } // namespace Test
} // namespace SushiBLAS
