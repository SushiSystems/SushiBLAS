/**************************************************************************/
/* symmetric_product.hpp                                                  */
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
 * @file symmetric_product.hpp
 * @brief Defines the SYCL symmetric matrix-vector product SBMV and SPMV share.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <SushiBLAS/support/launch_1d.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiBLAS/kernels/blas/conjugation.hpp>
#include "kernels/sycl/blas/level2/row_reduction.hpp"

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Computes y = alpha * A * x + beta * y for a symmetric or Hermitian A of
                 *        which one triangle is stored, per work-item or per work-group by row.
                 * @tparam Conj Unconjugated for a symmetric A, Conjugated for a Hermitian one.
                 * @tparam Access BandAccess or PackedAccess of the stored triangle.
                 * @param longest_row The most elements one row of A holds, both triangles counted.
                 * @see kernels/README.md
                 */
                template<typename T, typename Conj, typename Access>
                [[nodiscard]] sycl::event symmetric_product(sycl::queue& queue,
                                                            const Access& access, int64_t n,
                                                            int64_t longest_row,
                                                            T alpha, const T* a, const T* x,
                                                            int64_t incx, T beta, T* y,
                                                            int64_t incy,
                                                            const std::vector<sycl::event>& deps)
                {
                    using W = Accumulator<T>;

                    const Access mirror = access.transposed();
                    auto sym_at = [=](int64_t r, int64_t c) -> T
                    {
                        if (r == c)
                            return Conj::diagonal(access.at(a, r, r));
                        return access.holds(r, c) ? access.at(a, r, c)
                                                  : Conj::apply(access.at(a, c, r));
                    };
                    auto first_of = [=](int64_t i) -> int64_t
                    {
                        return std::min(access.first_column(i), mirror.first_column(i));
                    };
                    auto end_of = [=](int64_t i) -> int64_t
                    {
                        return std::max(access.end_column(i, n), mirror.end_column(i, n));
                    };

                    const std::size_t width = (n > 0)
                        ? row_reduction_width(Device::profile_for(queue), longest_row)
                        : std::size_t{0};

                    if (width != 0)
                    {
                        return queue.submit([&](sycl::handler& h)
                        {
                            h.depends_on(deps);
                            reduce_rows_by_group<W>(h, n, longest_row, width,
                                [=](int64_t i, int64_t term) -> W
                                {
                                    const int64_t j = first_of(i) + term;
                                    if (j >= end_of(i))
                                        return W(0);
                                    return multiply(W(sym_at(i, j)), W(x[j * incx]));
                                },
                                [=](int64_t i, W sum)
                                {
                                    y[i * incy] = (beta == T(0))
                                        ? T(multiply(W(alpha), sum))
                                        : T(multiply(W(alpha), sum) +
                                            multiply(W(beta), W(y[i * incy])));
                                });
                        });
                    }

                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        Launch::indexed(h, queue, static_cast<std::size_t>(n),
                            [=](std::size_t item)
                            {
                                const int64_t i = static_cast<int64_t>(item);
                                const int64_t end = end_of(i);

                                W sum(0);
                                for (int64_t j = first_of(i); j < end; ++j)
                                    sum += multiply(W(sym_at(i, j)), W(x[j * incx]));

                                y[i * incy] = (beta == T(0))
                                    ? T(multiply(W(alpha), sum))
                                    : T(multiply(W(alpha), sum) +
                                        multiply(W(beta), W(y[i * incy])));
                            });
                    });
                }
            } // namespace Detail
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
