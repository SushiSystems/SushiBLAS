/**************************************************************************/
/* tpsv.hpp                                                               */
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
 * @file tpsv.hpp
 * @brief Defines the SYCL kernel of TPSV.
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
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/blas_seam_kernels/packed_access.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/triangular_solve.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes TPSV: solves op(A) * x = b in place for a triangular matrix A
             *        packed in ap, where x holds b on entry, in the blocks a plan gives.
             * @param work The workspace Detail::triangular_solve states; null when none is needed.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event tpsv(sycl::queue& queue, const TriangularSolvePlan& plan,
                                           Core::Layout layout, Core::Uplo uplo,
                                           Core::Transpose trans, Core::Diag diag,
                                           int64_t n, const T* ap, T* x, int64_t incx,
                                           Accumulator<T>* work,
                                           const std::vector<sycl::event>& deps)
            {
                SB_LOG_DEBUG(K_BLAS_KERNELS_LOG_CATEGORY, "SYCL TPSV: {}x{} block={}", n, n, plan.block_width(n));

                const PackedAccess access = packed_access(layout, uplo, trans, n);
                const bool unit_diag = (diag == Core::Diag::Unit);
                return Detail::triangular_solve(queue, plan, access, access.is_upper(), unit_diag,
                                                n, n, ap, x, incx, work, deps);
            }

            /**
             * @brief Computes TPSV in the blocks the queue's device plans.
             * @param work The workspace Detail::triangular_solve states; null when none is needed.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event tpsv(sycl::queue& queue, Core::Layout layout,
                                           Core::Uplo uplo, Core::Transpose trans, Core::Diag diag,
                                           int64_t n, const T* ap, T* x, int64_t incx,
                                           Accumulator<T>* work,
                                           const std::vector<sycl::event>& deps)
            {
                const TriangularSolvePlan plan =
                    plan_vector_solve(Device::profile_for(queue), n, n);
                return tpsv<T>(queue, plan, layout, uplo, trans, diag, n, ap, x, incx, work, deps);
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
