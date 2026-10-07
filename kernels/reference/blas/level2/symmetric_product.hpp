/**************************************************************************/
/* symmetric_product.hpp                                                  */
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
 * @file symmetric_product.hpp
 * @brief Defines the reference symmetric matrix-vector product SBMV and SPMV share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <algorithm>
#include <SushiBLAS/kernels/blas/conjugation.hpp>
#include <SushiBLAS/support/reference_accumulator.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            namespace Detail
            {
                /**
                 * @brief Computes y = alpha * A * x + beta * y for a symmetric or Hermitian A of
                 *        which one triangle is stored.
                 * @tparam Conj Unconjugated for a symmetric A, Conjugated for a Hermitian one.
                 * @tparam Access BandAccess or PackedAccess of the stored triangle.
                 * @see kernels/README.md
                 */
                template<typename T, typename Conj, typename Access>
                void symmetric_product(const Access& access, int64_t n, T alpha, const T* a,
                                       const T* x, int64_t incx, T beta, T* y, int64_t incy)
                {
                    using RW = ReferenceAccumulator<T>;

                    const Access mirror = access.transposed();
                    auto sym_at = [&](int64_t r, int64_t c) -> T
                    {
                        if (r == c)
                            return Conj::diagonal(access.at(a, r, r));
                        return access.holds(r, c) ? access.at(a, r, c)
                                                  : Conj::apply(access.at(a, c, r));
                    };

                    for (int64_t i = 0; i < n; ++i)
                    {
                        const int64_t first =
                            std::min(access.first_column(i), mirror.first_column(i));
                        const int64_t end =
                            std::max(access.end_column(i, n), mirror.end_column(i, n));

                        ReferenceSum<T> sum;
                        for (int64_t j = first; j < end; ++j)
                            sum.add(RW(sym_at(i, j)) * RW(x[j * incx]));

                        y[i * incy] = (beta == T(0))
                                          ? T(RW(alpha) * sum.value())
                                          : T(RW(alpha) * sum.value() + RW(beta) * RW(y[i * incy]));
                    }
                }
            } // namespace Detail
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
