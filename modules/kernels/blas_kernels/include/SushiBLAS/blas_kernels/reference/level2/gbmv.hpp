/**************************************************************************/
/* gbmv.hpp                                                               */
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
 * @file gbmv.hpp
 * @brief Defines the reference kernel of GBMV.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas_seam_kernels/band_access.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes GBMV, y = alpha * op(A) * x + beta * y, for a band matrix A of m
             *        rows and n columns.
             * @param lda Elements stored per column of A (column-major) or per row (row-major).
             * @see kernels/README.md
             */
            template<typename T>
            void gbmv(Core::Layout layout, Core::Transpose trans, int64_t m, int64_t n, int64_t kl,
                      int64_t ku, T alpha, const T* a, int64_t lda, const T* x, int64_t incx,
                      T beta, T* y, int64_t incy)
            {
                using RW = ReferenceAccumulator<T>;

                const BandAccess access = band_access(layout, trans, kl, ku, lda);
                const bool transposed = Core::is_transposed(trans);
                const int64_t out_len = transposed ? n : m;
                const int64_t in_len = transposed ? m : n;

                for (int64_t o = 0; o < out_len; ++o)
                {
                    const int64_t end = access.end_column(o, in_len);

                    ReferenceSum<T> sum;
                    for (int64_t k = access.first_column(o); k < end; ++k)
                        sum.add(RW(access.at(a, o, k)) * RW(x[k * incx]));

                    y[o * incy] = (beta == T(0))
                                      ? T(RW(alpha) * sum.value())
                                      : T(RW(alpha) * sum.value() + RW(beta) * RW(y[o * incy]));
                }
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
