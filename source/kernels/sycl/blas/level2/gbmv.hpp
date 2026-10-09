/**************************************************************************/
/* gbmv.hpp                                                               */
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
 * @file gbmv.hpp
 * @brief Defines the SYCL kernel of GBMV.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/launch_1d.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiBLAS/kernels/blas/band_access.hpp>
#include "kernels/sycl/blas/level2/row_reduction.hpp"

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes GBMV, y = alpha * op(A) * x + beta * y, for a band matrix A of m
             *        rows and n columns, per work-item or per work-group by output element.
             * @param lda Elements stored per column of A (column-major) or per row (row-major).
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event gbmv(sycl::queue& queue, Core::Layout layout,
                                           Core::Transpose trans, int64_t m, int64_t n, int64_t kl,
                                           int64_t ku, T alpha, const T* a, int64_t lda,
                                           const T* x, int64_t incx, T beta, T* y, int64_t incy,
                                           const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;

                const BandAccess access = band_access(layout, trans, kl, ku, lda);
                const bool transposed = Core::is_transposed(trans);
                const int64_t out_len = transposed ? n : m;
                const int64_t in_len = transposed ? m : n;
                const int64_t longest_row = std::min(in_len, access.lower + access.upper + 1);

                SB_LOG_INFO("SYCL GBMV: {}x{}, kl={}, ku={}", m, n, kl, ku);

                // As in GEMV, the work-group shape serves only rows the storage keeps contiguous.
                const std::size_t width = (access.col == 1 && out_len > 0)
                    ? Detail::row_reduction_width(Device::profile_for(queue), longest_row)
                    : std::size_t{0};

                if (width != 0)
                {
                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        Detail::reduce_rows_by_group<W>(h, out_len, longest_row, width,
                            [=](int64_t o, int64_t term) -> W
                            {
                                const int64_t k = access.first_column(o) + term;
                                if (k >= access.end_column(o, in_len))
                                    return W(0);
                                return multiply(W(access.at(a, o, k)), W(x[k * incx]));
                            },
                            [=](int64_t o, W sum)
                            {
                                y[o * incy] = (beta == T(0))
                                    ? T(multiply(W(alpha), sum))
                                    : T(multiply(W(alpha), sum) +
                                        multiply(W(beta), W(y[o * incy])));
                            });
                    });
                }

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<std::size_t>(out_len),
                        [=](std::size_t item)
                        {
                            const int64_t o = static_cast<int64_t>(item);
                            const int64_t end = access.end_column(o, in_len);

                            W sum(0);
                            for (int64_t k = access.first_column(o); k < end; ++k)
                                sum += multiply(W(access.at(a, o, k)), W(x[k * incx]));

                            y[o * incy] = (beta == T(0))
                                ? T(multiply(W(alpha), sum))
                                : T(multiply(W(alpha), sum) + multiply(W(beta), W(y[o * incy])));
                        });
                });
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
