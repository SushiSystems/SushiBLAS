/**************************************************************************/
/* tbsv.hpp                                                               */
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
 * @file tbsv.hpp
 * @brief Defines the reference kernel of TBSV.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas_seam_kernels/band_access.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/triangular_solve.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes TBSV: solves op(A) * x = b in place for a triangular band matrix A
             *        with k off-diagonals, where x holds b on entry.
             * @param lda Elements stored per column of A (column-major) or per row (row-major).
             * @see kernels/README.md
             */
            template<typename T>
            void tbsv(Core::Layout layout, Core::Uplo uplo, Core::Transpose trans, Core::Diag diag,
                      int64_t n, int64_t k, const T* a, int64_t lda, T* x, int64_t incx)
            {
                const BandAccess access = triangle_band_access(layout, uplo, trans, k, lda);
                const bool op_upper = Core::is_transposed(trans) != (uplo == Core::Uplo::Upper);
                const bool unit_diag = (diag == Core::Diag::Unit);
                Detail::triangular_solve(access, op_upper, unit_diag, n, a, x, incx);
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
