/**************************************************************************/
/* symv.hpp                                                               */
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
 * @file symv.hpp
 * @brief Defines the SYCL kernel of SYMV.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/launch_1d.hpp>
#include "kernels/sycl/blas/level2/row_reduction.hpp"

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes SYMV, y = alpha*A*x + beta*y, reading only the requested triangle of A.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event symv(sycl::queue& queue, Core::Layout layout, bool upper,
                             int64_t n, T alpha, const T* a, int64_t lda,
                             const T* x, int64_t incx, T beta, T* y, int64_t incy,
                             const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;
                SB_LOG_INFO("SYCL SYMV: {}x{}", n, n);

                auto A_at = [=](int64_t r, int64_t c) -> T
                {
                    return (layout == Core::Layout::ROW_MAJOR) ? a[r * lda + c] : a[r + c * lda];
                };
                auto sym_at = [=](int64_t r, int64_t c) -> T
                {
                    if (upper) return (r <= c) ? A_at(r, c) : A_at(c, r);
                    return (r >= c) ? A_at(r, c) : A_at(c, r);
                };

                const std::size_t width = (n > 0)
                    ? Detail::row_reduction_width(Device::profile_for(queue), n)
                    : std::size_t{0};

                if (width != 0)
                {
                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        Detail::reduce_rows_by_group<W>(h, n, n, width,
                            [=](int64_t i, int64_t j) -> W
                            {
                                return W(sym_at(i, j)) * W(x[j * incx]);
                            },
                            [=](int64_t i, W sum)
                            {
                                y[i * incy] = (beta == T(0)) ? T(W(alpha) * sum)
                                                             : T(W(alpha) * sum + W(beta) * W(y[i * incy]));
                            });
                    });
                }

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<size_t>(n), [=](size_t k)
                    {
                        const int64_t i = static_cast<int64_t>(k);
                        W sum(0);
                        for (int64_t j = 0; j < n; ++j) sum += W(sym_at(i, j)) * W(x[j * incx]);
                        y[i * incy] = (beta == T(0)) ? T(W(alpha) * sum)
                                                     : T(W(alpha) * sum + W(beta) * W(y[i * incy]));
                    });
                });
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
