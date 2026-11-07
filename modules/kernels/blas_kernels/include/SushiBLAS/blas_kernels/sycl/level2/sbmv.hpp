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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file sbmv.hpp
 * @brief Defines the SYCL kernel of SBMV and HBMV.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/blas_kernels/blas_kernels_log_category.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas_seam_kernels/band_access.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/blas_kernels/matrix_vector_index.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/symmetric_product.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes SBMV, y = alpha * A * x + beta * y, for a symmetric band matrix A
             *        with k off-diagonals of which the uplo triangle is stored.
             * @tparam Conj Unconjugated for SBMV, Conjugated for HBMV.
             * @param lda Elements stored per column of A (column-major) or per row (row-major).
             * @see kernels/README.md
             */
            template<typename T, typename Conj = Unconjugated>
            [[nodiscard]] sycl::event sbmv(sycl::queue& queue, Core::Layout layout,
                                           Core::Uplo uplo, int64_t n, int64_t k, T alpha,
                                           const T* a, int64_t lda, const T* x, int64_t incx,
                                           T beta, T* y, int64_t incy,
                                           const std::vector<sycl::event>& deps)
            {
                const BandAccess access =
                    triangle_band_access(layout, uplo, Core::Transpose::NoTrans, k, lda);
                const int64_t longest_row = Indexing::band_row_length(n, k, k);
                return Detail::symmetric_product<T, Conj>(queue, access, n, longest_row, alpha, a,
                                                          x, incx, beta, y, incy, deps);
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
