/**************************************************************************/
/* spsymv.hpp                                                             */
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
 * @file spsymv.hpp
 * @brief Defines the SYCL kernel of the symmetric and Hermitian sparse matrix-vector product.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiBLAS/kernels/blas/conjugation.hpp>
#include "kernels/sparse/csr_pattern.hpp"
#include "kernels/sycl/sparse/row_blocks.hpp"

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes y = alpha * A * x + beta * y for an A given by one stored triangle.
             * @tparam Conj Unconjugated for a symmetric A, Conjugated for a Hermitian one.
             * @param uplo The triangle of pattern that is read; entries of the other are skipped.
             * @param partials blocks * rows elements of device scratch.
             * @param blocks The value scatter_blocks returned for rows outputs.
             * @see kernels/README.md
             */
            template<typename T, typename Index, typename Conj = Unconjugated>
            [[nodiscard]] sycl::event spsymv(sycl::queue& queue, Core::Uplo uplo,
                                             CsrPattern<Index> pattern, const T* values, T alpha,
                                             const T* x, int64_t incx, T beta, T* y, int64_t incy,
                                             Accumulator<T>* partials, std::size_t blocks,
                                             const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;

                SB_LOG_INFO("SYCL SPSYMV: order {}, {} entries", pattern.rows, pattern.nnz);
                SB_ASSERT(pattern.rows == 0 || (partials != nullptr && blocks > 0),
                          "A symmetric spmv needs the partial sums its caller sized");

                const int64_t n = pattern.rows;
                const bool lower = (uplo == Core::Uplo::Lower);
                const int64_t used_blocks = static_cast<int64_t>(std::max<std::size_t>(blocks, 1));

                // Block b alone writes partials[b * n, (b + 1) * n).
                const sycl::event scattered = queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.parallel_for(sycl::range<1>(static_cast<std::size_t>(used_blocks)),
                                   [=](sycl::id<1> id)
                    {
                        const int64_t block = static_cast<int64_t>(id[0]);
                        W* partial = partials + block * n;

                        for (int64_t j = 0; j < n; ++j)
                            partial[j] = W(0);

                        const int64_t first = Detail::block_first_row(pattern, block, used_blocks);
                        const int64_t last = Detail::block_end_row(pattern, block, used_blocks);
                        for (int64_t i = first; i < last; ++i)
                        {
                            const W xi = W(x[i * incx]);
                            for (int64_t k = pattern.row_begin(i); k < pattern.row_end(i); ++k)
                            {
                                const int64_t j = pattern.column(k);
                                const W value = W(values[k]);
                                if (j == i)
                                {
                                    partial[i] += multiply(Conj::diagonal(value), xi);
                                }
                                else if (lower ? j < i : j > i)
                                {
                                    partial[i] += multiply(value, W(x[j * incx]));
                                    partial[j] += multiply(Conj::apply(value), xi);
                                }
                            }
                        }
                    });
                });

                return Detail::fold_blocks<T>(queue, scattered, n, used_blocks, partials, alpha,
                                              beta, y, incy);
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
