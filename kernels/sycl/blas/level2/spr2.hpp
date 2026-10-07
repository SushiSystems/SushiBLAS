/**************************************************************************/
/* spr2.hpp                                                               */
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
 * @file spr2.hpp
 * @brief Defines the SYCL kernel of SPR2 and HPR2.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/kernels/blas/conjugation.hpp>
#include <SushiBLAS/kernels/blas/packed_access.hpp>
#include "kernels/sycl/blas/level2/syr2.hpp"

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes SPR2, A += alpha * (x * y^T + y * x^T), on the uplo triangle of A
             *        packed in ap, one work-item per element of the square.
             * @tparam Conj Unconjugated for SPR2, Conjugated for HPR2, which adds
             *         alpha * x * y^H + conj(alpha) * y * x^H.
             * @see kernels/README.md
             */
            template<typename T, typename Conj = Unconjugated>
            [[nodiscard]] sycl::event spr2(sycl::queue& queue, Core::Layout layout,
                                           Core::Uplo uplo, int64_t n, T alpha, const T* x,
                                           int64_t incx, const T* y, int64_t incy, T* ap,
                                           const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;

                SB_LOG_INFO("SYCL SPR2: {}x{}", n, n);

                const PackedAccess access =
                    packed_access(layout, uplo, Core::Transpose::NoTrans, n);

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.parallel_for(
                        sycl::range<2>(static_cast<std::size_t>(n), static_cast<std::size_t>(n)),
                        [=](sycl::id<2> idx)
                        {
                            const int64_t i = static_cast<int64_t>(idx[0]);
                            const int64_t j = static_cast<int64_t>(idx[1]);
                            if (!access.holds(i, j))
                                return;

                            T& ref = ap[access.index(i, j)];
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
