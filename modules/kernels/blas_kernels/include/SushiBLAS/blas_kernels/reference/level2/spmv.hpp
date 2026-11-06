/**************************************************************************/
/* spmv.hpp                                                               */
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
 * @file spmv.hpp
 * @brief Defines the reference kernel of SPMV and HPMV.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/blas_seam_kernels/packed_access.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/symmetric_product.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes SPMV, y = alpha * A * x + beta * y, for a symmetric matrix A of
             *        which the uplo triangle is packed in ap.
             * @tparam Conj Unconjugated for SPMV, Conjugated for HPMV.
             * @see kernels/README.md
             */
            template<typename T, typename Conj = Unconjugated>
            void spmv(Core::Layout layout, Core::Uplo uplo, int64_t n, T alpha, const T* ap,
                      const T* x, int64_t incx, T beta, T* y, int64_t incy)
            {
                const PackedAccess access =
                    packed_access(layout, uplo, Core::Transpose::NoTrans, n);
                Detail::symmetric_product<T, Conj>(access, n, alpha, ap, x, incx, beta, y, incy);
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
