/**************************************************************************/
/* syr2.hpp                                                               */
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
 * @file syr2.hpp
 * @brief Defines the SYCL kernel of SYR2 and HER2.
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
#include <SushiBLAS/kernels/blas/conjugation.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /** @brief Returns the SYR2 term alpha * (x_i * y_j + y_i * x_j). */
                template<typename W>
                [[nodiscard]] inline W rank2_term(Unconjugated, W alpha, W xi, W yj, W yi, W xj)
                {
                    return multiply(alpha, multiply(xi, yj) + multiply(yi, xj));
                }

                /** @brief Returns the HER2 term, which conjugates y_j, x_j and the second alpha. */
                template<typename W>
                [[nodiscard]] inline W rank2_term(Conjugated, W alpha, W xi, W yj, W yi, W xj)
                {
                    return multiply(multiply(alpha, xi), Conjugated::apply(yj)) +
                           multiply(multiply(Conjugated::apply(alpha), yi), Conjugated::apply(xj));
                }
            } // namespace Detail

            /**
             * @brief Computes SYR2 or HER2, A(i, j) += rank2_term, in the triangle uplo names.
             * @tparam Conj Unconjugated (syr2) or Conjugated (her2, which stores a real diagonal).
             * @see kernels/README.md
             */
            template<typename T, typename Conj = Unconjugated>
            [[nodiscard]] sycl::event syr2(sycl::queue& queue, Core::Layout layout, Core::Uplo uplo,
                             int64_t n, T alpha, const T* x, int64_t incx,
                             const T* y, int64_t incy, T* a, int64_t lda,
                             const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;
                const bool upper = (uplo == Core::Uplo::UPPER);
                SB_LOG_INFO("SYCL SYR2: {}x{}", n, n);

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    const sycl::range<2> elements(static_cast<size_t>(n), static_cast<size_t>(n));
                    h.parallel_for(elements, [=](sycl::id<2> idx)
                    {
                        int64_t i = idx[0], j = idx[1];
                        bool in_triangle = upper ? (j >= i) : (j <= i);
                        if (!in_triangle)
                            return;
                        T& ref = (layout == Core::Layout::ROW_MAJOR) ? a[i * lda + j]
                                                                     : a[i + j * lda];
                        const W term = Detail::rank2_term(Conj{}, W(alpha),
                                                          W(x[i * incx]), W(y[j * incy]),
                                                          W(y[i * incy]), W(x[j * incx]));
                        const W updated = W(ref) + term;
                        ref = T((i == j) ? Conj::diagonal(updated) : updated);
                    });
                });
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
