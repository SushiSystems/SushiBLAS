/**************************************************************************/
/* symv.hpp                                                               */
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
 * @file symv.hpp
 * @brief Defines the reference kernel of SYMV and HEMV.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes SYMV or HEMV, y = alpha*A*x + beta*y, each element in a ReferenceSum.
             * @tparam Conj Unconjugated for symv, Conjugated for hemv.
             * @param a Only the triangle uplo names is read.
             */
            template<typename T, typename Conj = Unconjugated>
            void symv(Core::Layout layout, Core::Uplo uplo, int64_t n,
                      T alpha, const T* a, int64_t lda,
                      const T* x, int64_t incx,
                      T beta, T* y, int64_t incy)
            {
                using RW = ReferenceAccumulator<T>;

                auto A_at = [&](int64_t r, int64_t c) -> T
                {
                    return (layout == Core::Layout::RowMajor) ? a[r * lda + c] : a[r + c * lda];
                };
                const bool upper = (uplo == Core::Uplo::Upper);
                auto sym_at = [&](int64_t r, int64_t c) -> T
                {
                    if (r == c)
                        return Conj::diagonal(A_at(r, r));
                    const bool stored = upper ? (r < c) : (r > c);
                    return stored ? A_at(r, c) : Conj::apply(A_at(c, r));
                };

                for (int64_t i = 0; i < n; ++i)
                {
                    ReferenceSum<T> sum;
                    for (int64_t j = 0; j < n; ++j)
                        sum.add(RW(sym_at(i, j)) * RW(x[j * incx]));

                    y[i * incy] = (beta == T(0))
                        ? T(RW(alpha) * sum.value())
                        : T(RW(alpha) * sum.value() + RW(beta) * RW(y[i * incy]));
                }
            }

        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
