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
            /**
             * @brief Computes SYR or HER, A += alpha * x * Conj(x)^T, one work-item per element of
             *        the triangle uplo names.
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
                SB_LOG_DEBUG(K_BLAS_KERNELS_LOG_CATEGORY, "SYCL SYR: {}x{}", n, n);

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
                        const W updated = W(ref) + multiply(multiply(WS(alpha), W(x[i * incx])),
                                                            Conj::apply(W(x[j * incx])));
                        ref = T((i == j) ? Conj::diagonal(updated) : updated);
                    });
                });
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
