/**************************************************************************/
/* syr.hpp                                                                */
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
 * @file syr.hpp
 * @brief Defines the SYCL kernel of SYR and HER.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <type_traits>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes SYR or HER, A += alpha * x * Conj(x)^T, in the triangle uplo names.
             * @tparam Conj Unconjugated (syr) or Conjugated (her, which stores a real diagonal).
             * @tparam Scalar The type of alpha: T for syr, the real component type of T for her.
             * @see kernels/README.md
             */
            template<typename T, typename Conj = Unconjugated, typename Scalar = T>
            [[nodiscard]] sycl::event syr(sycl::queue& queue, Core::Layout layout, Core::Uplo uplo,
                            int64_t n, Scalar alpha, const T* x, int64_t incx,
                            T* a, int64_t lda, const std::vector<sycl::event>& deps)
            {
                static_assert(std::is_same_v<Scalar, T> || std::is_same_v<Scalar, RealType<T>>,
                              "alpha is a T for SYR and the real component type of T for HER.");

                using W = Accumulator<T>;
                using WS = Accumulator<Scalar>;
                const bool upper = (uplo == Core::Uplo::Upper);
                SB_LOG_INFO("SYCL SYR: {}x{}", n, n);

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
                        T& ref = (layout == Core::Layout::RowMajor) ? a[i * lda + j]
                                                                     : a[i + j * lda];
                        const W updated = W(ref) + multiply(multiply(WS(alpha), W(x[i * incx])),
                                                            Conj::apply(W(x[j * incx])));
                        ref = T((i == j) ? Conj::diagonal(updated) : updated);
                    });
                });
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
