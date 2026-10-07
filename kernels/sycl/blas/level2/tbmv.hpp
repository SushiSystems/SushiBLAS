/**************************************************************************/
/* tbmv.hpp                                                               */
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
 * @file tbmv.hpp
 * @brief Defines the SYCL kernel of TBMV.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/kernels/blas/band_access.hpp>
#include "kernels/sycl/blas/level2/triangular_product.hpp"

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes TBMV, x = op(A) * x, in place for a triangular band matrix A with k
             *        off-diagonals.
             * @param lda Elements stored per column of A (column-major) or per row (row-major).
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event tbmv(sycl::queue& queue, Core::Layout layout,
                                           Core::Uplo uplo, Core::Transpose trans, Core::Diag diag,
                                           int64_t n, int64_t k, const T* a, int64_t lda, T* x,
                                           int64_t incx, const std::vector<sycl::event>& deps)
            {
                SB_LOG_INFO("SYCL TBMV: {}x{}, k={}", n, n, k);

                const BandAccess access = triangle_band_access(layout, uplo, trans, k, lda);
                const bool op_upper = Core::is_transposed(trans) != (uplo == Core::Uplo::UPPER);
                const bool unit_diag = (diag == Core::Diag::UNIT);
                return Detail::triangular_product(queue, access, op_upper, unit_diag, n, a, x,
                                                  incx, deps);
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
