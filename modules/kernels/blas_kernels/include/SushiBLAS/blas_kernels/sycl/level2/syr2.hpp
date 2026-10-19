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
#include <SushiBLAS/blas_kernels/blas_kernels_log_category.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/blas_seam_kernels/packed_access.hpp>

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
             * @brief Computes SYR2 or HER2, A(i, j) += rank2_term, one work-item per element of the
             *        triangle uplo names.
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
                SB_LOG_DEBUG(K_BLAS_KERNELS_LOG_CATEGORY, "SYCL SYR2: {}x{}", n, n);

                // The walk of the triangle runs its lines along the contiguous axis of A.
                const PackedAccess walk = packed_access(layout, uplo, Core::Transpose::NoTrans, n);
                const std::size_t stored = static_cast<std::size_t>(n * (n + 1) / 2);

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, stored, [=](std::size_t t)
                    {
                        const MatrixPosition at = walk.position(static_cast<int64_t>(t));
                        const int64_t i = at.row;
                        const int64_t j = at.col;
                        T& ref = (layout == Core::Layout::RowMajor) ? a[i * lda + j]
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
