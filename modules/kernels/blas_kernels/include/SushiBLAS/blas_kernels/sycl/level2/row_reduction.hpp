/**************************************************************************/
/* row_reduction.hpp                                                      */
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
 * @file row_reduction.hpp
 * @brief Defines the work-group-per-row reduction GEMV, SYMV and TRMV share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <sycl/sycl.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Returns the work-group width that reduces one row of length len, or 0 for one work-item per row.
                 * @return A power of two the device accepts, or 0 when the device profile keeps the per-item shape.
                 * @see kernels/README.md
                 */
                [[nodiscard]] inline std::size_t row_reduction_width(const Device::Profile& p,
                                                                     int64_t len) noexcept
                {
                    if (len <= 0)
                        return 0;
                    if (Reduce::planned_partials(p, static_cast<std::size_t>(len)) == 0)
                        return 0;
                    return Reduce::group_width(p);
                }

                /**
                 * @brief Holds the columns one row folds: first up to, and not including, last.
                 * @see kernels/README.md
                 */
                struct RowWindow
                {
                    int64_t first = 0; /**< The first column folded. */
                    int64_t last = 0;  /**< One past the last column folded. */
                };

                /**
                 * @brief Gives every row the same window, the whole row.
                 * @see kernels/README.md
                 */
                struct WholeRow
                {
                    int64_t length = 0; /**< Columns in a row. */

                    /** @brief Returns the window of a row: every column. */
                    [[nodiscard]] constexpr RowWindow operator()(int64_t) const noexcept
                    {
                        return RowWindow{0, length};
                    }
                };

                /**
                 * @brief Gives row i of a triangular matrix its stored columns, diagonal included.
                 * @see kernels/README.md
                 */
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

                /**
                 * @brief Launches one work-group per row; each folds `element(row, k)` over the
                 *        row's window in a fixed-order tree.
                 * @tparam T The partial type: Accumulator of the element type.
                 * @tparam Window Callable `(int64_t row) -> RowWindow`.
                 * @pre width is a power of two from row_reduction_width, and rows is above zero.
                 * @see kernels/README.md
                 */
                template<typename T, typename Window, typename Element, typename Store>
                void reduce_row_windows_by_group(sycl::handler& h, int64_t rows, std::size_t width,
                                                 Window window, Element element, Store store)
                {
                    sycl::local_accessor<T, 1> tile(sycl::range<1>(width), h);
                    const std::size_t global = static_cast<std::size_t>(rows) * width;

                    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(width)),
                                   [=](sycl::nd_item<1> it)
                    {
                        const int64_t row = static_cast<int64_t>(it.get_group(0));
                        const std::size_t lid = it.get_local_id(0);
                        const int64_t step = static_cast<int64_t>(width);
                        const RowWindow columns = window(row);

                        T sum(0);
                        const int64_t lane = static_cast<int64_t>(lid);
                        for (int64_t k = columns.first + lane; k < columns.last; k += step)
                            sum += element(row, k);
                        tile[lid] = sum;

                        sycl::group_barrier(it.get_group());

                        // Sub-group 0 takes the last levels when a power-of-two sub-group fits.
                        const sycl::sub_group sg = it.get_sub_group();
                        const std::size_t lanes = sg.get_max_local_range()[0];
                        const bool shuffles =
                            lanes > 1 && (lanes & (lanes - 1)) == 0 && width >= lanes;
                        const std::size_t floor = shuffles ? lanes : std::size_t{1};

                        // width is a power of two, so the halving tree pairs every lane in a fixed order.
                        for (std::size_t off = width / 2; off >= floor; off >>= 1)
                        {
                            if (lid < off)
                                tile[lid] = tile[lid] + tile[lid + off];
                            sycl::group_barrier(it.get_group());
                        }

                        if (!shuffles)
                        {
                            if (lid == 0)
                                store(row, tile[0]);
                            return;
                        }

                        // The shuffles pair lane l with lane l + off, the pairs the tree takes.
                        if (sg.get_group_linear_id() != 0)
                            return;
                        const std::size_t l = sg.get_local_linear_id();
                        T value = tile[l];
                        for (std::size_t off = lanes / 2; off > 0; off >>= 1)
                        {
                            const T other = sycl::shift_group_left(sg, value, off);
                            if (l < off)
                                value = value + other;
                        }
                        if (l == 0)
                            store(row, value);
                    });
                }

                /**
                 * @brief Launches one work-group per row; each folds `element(row, k)` over k in a fixed-order tree.
                 * @tparam T The partial type: Accumulator of the element type.
                 * @tparam Element Callable `(int64_t row, int64_t k) -> T`.
                 * @tparam Store Callable `(int64_t row, T sum)`, called once per row by local item 0.
                 * @pre width is a power of two returned by row_reduction_width.
                 * @see kernels/README.md
                 */
                template<typename T, typename Element, typename Store>
                void reduce_rows_by_group(sycl::handler& h, int64_t rows, int64_t len,
                                          std::size_t width, Element element, Store store)
                {
                    reduce_row_windows_by_group<T>(h, rows, width, WholeRow{len},
                                                   std::move(element), std::move(store));
                }
            } // namespace Detail
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
