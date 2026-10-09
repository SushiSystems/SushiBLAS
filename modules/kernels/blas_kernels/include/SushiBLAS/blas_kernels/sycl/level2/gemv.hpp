/**************************************************************************/
/* gemv.hpp                                                               */
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
 * @file gemv.hpp
 * @brief Defines the SYCL kernel of GEMV.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiBLAS/support/launch_1d.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/row_reduction.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes GEMV, y = alpha*op(A)*x + beta*y, one output element per work-item
             *        or per work-group.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event gemv(sycl::queue& queue, Core::Layout layout,
                                           Core::Transpose trans, int64_t m, int64_t n, T alpha,
                                           const T* a, int64_t lda, const T* x, int64_t incx,
                                           T beta, T* y, int64_t incy,
                                           const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;
                const OperandAccess access = operand_access(layout, trans, lda);
                const bool transposed = Core::is_transposed(trans);
                const int64_t out_len = transposed ? n : m;
                const int64_t in_len = transposed ? m : n;

                SB_LOG_INFO("SYCL GEMV: {}x{}", m, n);

                const bool row_contiguous = (layout == Core::Layout::RowMajor) != transposed;
                const std::size_t width = (row_contiguous && out_len > 0)
                    ? Detail::row_reduction_width(Device::profile_for(queue), in_len)
                    : std::size_t{0};

                if (width != 0)
                {
                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        // A row-contiguous op(A) holds (o, k) at a[o * lda + k] in both layouts.
                        Detail::reduce_rows_by_group<W>(h, out_len, in_len, width,
                            [=](int64_t o, int64_t k) -> W
                            {
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
                    Launch::indexed(h, queue, static_cast<size_t>(out_len), [=](size_t k)
                    {
                        const int64_t o = static_cast<int64_t>(k);
                        W sum(0);
                        for (int64_t k = 0; k < in_len; ++k)
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
