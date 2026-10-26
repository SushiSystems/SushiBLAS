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
 * @brief Defines the SYCL kernel of TBSV.
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
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/blas_seam_kernels/band_access.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/triangular_solve.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes TBSV: solves op(A) * x = b in place for a triangular band matrix A
             *        with k off-diagonals, where x holds b on entry, in the blocks a plan gives.
             * @param lda Elements stored per column of A (column-major) or per row (row-major).
             * @param work The workspace Detail::triangular_solve states; null when none is needed.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] Device::SubmittedEvents tbsv(sycl::queue& queue, const TriangularSolvePlan& plan,
                                           Core::Layout layout, Core::Uplo uplo,
                                           Core::Transpose trans, Core::Diag diag,
                                           int64_t n, int64_t k, const T* a, int64_t lda, T* x,
                                           int64_t incx, Accumulator<T>* work,
                                           const std::vector<sycl::event>& deps)
            {
                const BandAccess access = triangle_band_access(layout, uplo, trans, k, lda);
                const bool op_upper = Core::is_transposed(trans) != (uplo == Core::Uplo::Upper);
                const bool unit_diag = (diag == Core::Diag::Unit);
                return Detail::triangular_solve(queue, plan, access, op_upper, unit_diag, n, k, a,
                                                x, incx, work, deps);
            }

            /**
             * @brief Computes TBSV in the blocks the queue's device plans.
             * @param lda Elements stored per column of A (column-major) or per row (row-major).
             * @param work The workspace Detail::triangular_solve states; null when none is needed.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] Device::SubmittedEvents tbsv(sycl::queue& queue, Core::Layout layout,
                                           Core::Uplo uplo, Core::Transpose trans, Core::Diag diag,
                                           int64_t n, int64_t k, const T* a, int64_t lda, T* x,
                                           int64_t incx, Accumulator<T>* work,
                                           const std::vector<sycl::event>& deps)
            {
                const TriangularSolvePlan plan =
                    plan_vector_solve(Device::profile_for(queue), n, k);
                return tbsv<T>(queue, plan, layout, uplo, trans, diag, n, k, a, lda, x, incx, work,
                               deps);
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
