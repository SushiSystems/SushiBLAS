/**************************************************************************/
/* spr.hpp                                                                */
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
 * @file spr.hpp
 * @brief Defines the SYCL kernel of SPR and HPR.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/blas_seam_kernels/packed_access.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes SPR, A += alpha * x * x^T, on the uplo triangle of A packed in ap,
             *        one work-item per element of the square.
             * @tparam Conj Unconjugated for SPR, Conjugated for HPR, which adds alpha * x * x^H.
             * @tparam Scalar T for SPR, the real component type of T for HPR.
             * @see kernels/README.md
             */
            template<typename T, typename Conj = Unconjugated, typename Scalar = T>
            [[nodiscard]] sycl::event spr(sycl::queue& queue, Core::Layout layout, Core::Uplo uplo,
                                          int64_t n, Scalar alpha, const T* x, int64_t incx,
                                          T* ap, const std::vector<sycl::event>& deps)
            {
                static_assert(std::is_same_v<Scalar, T> || std::is_same_v<Scalar, RealType<T>>,
                              "alpha is a T for SPR and the real component type of T for HPR.");

                using W = Accumulator<T>;
                using WS = Accumulator<Scalar>;

                SB_LOG_INFO("SYCL SPR: {}x{}", n, n);

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
                            const W updated =
                                W(ref) + multiply(multiply(WS(alpha), W(x[i * incx])),
                                                  Conj::apply(W(x[j * incx])));
                            ref = T((i == j) ? Conj::diagonal(updated) : updated);
                        });
                });
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
