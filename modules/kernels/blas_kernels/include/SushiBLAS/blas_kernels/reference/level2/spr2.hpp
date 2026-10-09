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
 * @brief Defines the reference kernel of SPR2 and HPR2.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/blas_seam_kernels/row_cursor.hpp>
#include <SushiBLAS/blas_seam_kernels/packed_access.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/syr2.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes SPR2, A += alpha * (x * y^T + y * x^T), on the uplo triangle of A
             *        packed in ap.
             * @tparam Conj Unconjugated for SPR2, Conjugated for HPR2, which adds
             *         alpha * x * y^H + conj(alpha) * y * x^H.
             * @see kernels/README.md
             */
            template<typename T, typename Conj = Unconjugated>
            void spr2(Core::Layout layout, Core::Uplo uplo, int64_t n, T alpha, const T* x,
                      int64_t incx, const T* y, int64_t incy, T* ap)
            {
                using W = Accumulator<T>;

                const PackedAccess access =
                    packed_access(layout, uplo, Core::Transpose::NoTrans, n);

                for (int64_t i = 0; i < n; ++i)
                {
                    const int64_t end = access.end_column(i, n);
                    const int64_t first = access.first_column(i);
                    RowCursor cursor = access.row_cursor(i, first);
                    for (int64_t j = first; j < end; ++j, cursor.advance())
                    {
                        T& ref = ap[cursor.offset];
                        const W term = Detail::rank2_term(Conj{}, W(alpha),
                                                          W(x[i * incx]), W(y[j * incy]),
                                                          W(y[i * incy]), W(x[j * incx]));
                        const W updated = W(ref) + term;
                        ref = T((i == j) ? Conj::diagonal(updated) : updated);
                    }
                }
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
