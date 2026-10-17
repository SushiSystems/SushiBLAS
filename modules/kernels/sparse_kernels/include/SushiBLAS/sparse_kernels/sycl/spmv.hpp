/**************************************************************************/
/* spmv.hpp                                                               */
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
 * @file spmv.hpp
 * @brief Defines the SYCL kernel of the sparse matrix-vector product.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/sparse_kernels/sparse_kernels_log_category.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/sycl/row_blocks.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes y = alpha * op(A) * x + beta * y for a CSR matrix A.
             * @param beta Exactly zero skips reading y.
             * @param partials blocks * cols elements of device scratch for a transposed product.
             * @param blocks The value scatter_blocks returned for cols outputs.
             * @note A transposed result may differ in the last bits from a row-by-row sum.
             * @see kernels/README.md
             */
            template<typename T, typename Index>
            [[nodiscard]] sycl::event spmv(sycl::queue& queue, Core::Transpose trans,
                                           CsrPattern<Index> pattern, const T* values, T alpha,
                                           const T* x, int64_t incx, T beta, T* y, int64_t incy,
                                           Accumulator<T>* partials, std::size_t blocks,
                                           const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;

                SB_LOG_DEBUG(K_SPARSE_KERNELS_LOG_CATEGORY, "SYCL SPARSE SPMV: {}x{}, {} entries", pattern.rows, pattern.cols,
                            pattern.nnz);

                if (!Core::is_transposed(trans))
                {
                    const bool keeps_y = !(beta == T(0));

                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        Launch::indexed(h, queue, static_cast<std::size_t>(pattern.rows),
                                        [=](std::size_t index)
                        {
                            const int64_t i = static_cast<int64_t>(index);

                            W sum(0);
                            for (int64_t k = pattern.row_begin(i); k < pattern.row_end(i); ++k)
                                sum += multiply(W(values[k]), W(x[pattern.column(k) * incx]));

                            const W scaled = multiply(W(alpha), sum);
                            const int64_t out = i * incy;
                            y[out] = keeps_y ? static_cast<T>(scaled + multiply(W(beta), W(y[out])))
                                             : static_cast<T>(scaled);
                        });
                    });
                }

                SB_ASSERT(pattern.cols == 0 || (partials != nullptr && blocks > 0),
                          "A transposed spmv needs the partial sums its caller sized");

                const bool conjugated = Core::is_conjugated(trans);
                const int64_t outputs = pattern.cols;
                const int64_t used_blocks = static_cast<int64_t>(std::max<std::size_t>(blocks, 1));

                // Block b alone writes partials[b * outputs, (b + 1) * outputs).
                const sycl::event scattered = queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.parallel_for(sycl::range<1>(static_cast<std::size_t>(used_blocks)),
                                   [=](sycl::id<1> id)
                    {
                        const int64_t block = static_cast<int64_t>(id[0]);
                        W* partial = partials + block * outputs;

                        for (int64_t j = 0; j < outputs; ++j)
                            partial[j] = W(0);

                        const int64_t first = Detail::block_first_row(pattern, block, used_blocks);
                        const int64_t last = Detail::block_end_row(pattern, block, used_blocks);
                        for (int64_t i = first; i < last; ++i)
                        {
                            const W xi = W(x[i * incx]);
                            for (int64_t k = pattern.row_begin(i); k < pattern.row_end(i); ++k)
                                partial[pattern.column(k)] +=
                                    multiply(W(conjugate_if(conjugated, values[k])), xi);
                        }
                    });
                });

                return Detail::fold_blocks<T>(queue, scattered, outputs, used_blocks, partials,
                                              alpha, beta, y, incy);
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
