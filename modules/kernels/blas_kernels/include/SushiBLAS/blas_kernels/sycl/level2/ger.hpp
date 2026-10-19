/**************************************************************************/
/* ger.hpp                                                                */
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
 * @file ger.hpp
 * @brief Defines the SYCL kernel of GER, GERU and GERC.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/blas_kernels/blas_kernels_log_category.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes GER, A += alpha * x * Conj(y)^T, one work-item per element of A,
             *        neighbouring work-items on neighbouring elements.
             * @tparam Conj Unconjugated for ger and geru, Conjugated for gerc.
             * @see kernels/README.md
             */
            template<typename T, typename Conj = Unconjugated>
            [[nodiscard]] sycl::event ger(sycl::queue& queue, Core::Layout layout,
                            int64_t m, int64_t n, T alpha,
                            const T* x, int64_t incx, const T* y, int64_t incy,
                            T* a, int64_t lda, const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;
                SB_LOG_DEBUG(K_BLAS_KERNELS_LOG_CATEGORY, "SYCL GER: {}x{}", m, n);

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    // The fastest index walks the axis the elements of A are adjacent on.
                    const bool by_columns = (layout == Core::Layout::ColumnMajor);
                    const std::size_t rows = static_cast<std::size_t>(m);
                    const std::size_t cols = static_cast<std::size_t>(n);
                    const sycl::range<2> elements = by_columns ? sycl::range<2>(cols, rows)
                                                               : sycl::range<2>(rows, cols);
                    h.parallel_for(elements, [=](sycl::id<2> idx)
                    {
                        const int64_t i = static_cast<int64_t>(by_columns ? idx[1] : idx[0]);
                        const int64_t j = static_cast<int64_t>(by_columns ? idx[0] : idx[1]);
                        T& ref = (layout == Core::Layout::RowMajor) ? a[i * lda + j]
                                                                     : a[i + j * lda];
                        ref = T(W(ref) + multiply(multiply(W(alpha), W(x[i * incx])),
                                                  Conj::apply(W(y[j * incy]))));
                    });
                });
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
