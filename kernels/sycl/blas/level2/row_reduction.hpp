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
 * @brief Defines the work-group-per-row reduction GEMV and SYMV share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/support/reduce_1d.hpp>

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
                    if (len <= 0) return 0;
                    if (Reduce::planned_partials(p, static_cast<std::size_t>(len)) == 0) return 0;
                    return Reduce::group_width(p);
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
                    sycl::local_accessor<T, 1> tile(sycl::range<1>(width), h);
                    const std::size_t global = static_cast<std::size_t>(rows) * width;

                    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(width)),
                                   [=](sycl::nd_item<1> it)
                    {
                        const int64_t row = static_cast<int64_t>(it.get_group(0));
                        const std::size_t lid = it.get_local_id(0);
                        const int64_t step = static_cast<int64_t>(width);

                        T sum(0);
                        for (int64_t k = static_cast<int64_t>(lid); k < len; k += step)
                            sum += element(row, k);
                        tile[lid] = sum;

                        sycl::group_barrier(it.get_group());

                        // width is a power of two, so the halving tree pairs every lane in a fixed order.
                        for (std::size_t off = width / 2; off > 0; off >>= 1)
                        {
                            if (lid < off) tile[lid] = tile[lid] + tile[lid + off];
                            sycl::group_barrier(it.get_group());
                        }

                        if (lid == 0) store(row, tile[0]);
                    });
                }
            } // namespace Detail
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
