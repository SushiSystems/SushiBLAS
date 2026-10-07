/**************************************************************************/
/* sbmv.hpp                                                               */
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
 * @file sbmv.hpp
 * @brief Defines the reference kernel of SBMV and HBMV.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/kernels/blas/band_access.hpp>
#include <SushiBLAS/kernels/blas/conjugation.hpp>
#include "kernels/reference/blas/level2/symmetric_product.hpp"

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes SBMV, y = alpha * A * x + beta * y, for a symmetric band matrix A
             *        with k off-diagonals of which the uplo triangle is stored.
             * @tparam Conj Unconjugated for SBMV, Conjugated for HBMV.
             * @param lda Elements stored per column of A (column-major) or per row (row-major).
             * @see kernels/README.md
             */
            template<typename T, typename Conj = Unconjugated>
            void sbmv(Core::Layout layout, Core::Uplo uplo, int64_t n, int64_t k, T alpha,
                      const T* a, int64_t lda, const T* x, int64_t incx, T beta, T* y,
                      int64_t incy)
            {
                const BandAccess access =
                    triangle_band_access(layout, uplo, Core::Transpose::NO_TRANS, k, lda);
                Detail::symmetric_product<T, Conj>(access, n, alpha, a, x, incx, beta, y, incy);
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
