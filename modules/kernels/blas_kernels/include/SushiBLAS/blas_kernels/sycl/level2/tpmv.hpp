/**************************************************************************/
/* tpmv.hpp                                                               */
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
 * @file tpmv.hpp
 * @brief Defines the SYCL kernel of TPMV.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/submitted_events.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/blas_kernels/blas_kernels_log_category.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas_seam_kernels/packed_access.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/triangular_product.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes TPMV, x = op(A) * x, in place for a triangular matrix A packed in
             *        ap.
             * @param work Device scratch of n elements that no other task touches; may be null when
             *        n is 0.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] Device::SubmittedEvents tpmv(sycl::queue& queue, Core::Layout layout,
                                           Core::Uplo uplo, Core::Transpose trans, Core::Diag diag,
                                           int64_t n, const T* ap, T* x, int64_t incx, T* work,
                                           const std::vector<sycl::event>& deps)
            {
                const PackedAccess access = packed_access(layout, uplo, trans, n);
                const bool unit_diag = (diag == Core::Diag::Unit);
                return Detail::triangular_product(queue, access, unit_diag, n, ap, x, incx, work,
                                                  deps);
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
