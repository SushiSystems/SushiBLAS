/**************************************************************************/
/* gemm_accessed_packed.hpp                                               */
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
 * @file gemm_accessed_packed.hpp
 * @brief Declares the form of gemm_accessed that packs its operands where the profile selects it.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_epilogue.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/packed_product_plan.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm_packed.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes what gemm_accessed without scratch computes, and packs the operands
             *        when the device profile selects the packed kernel.
             * @tparam AccessA Brick with `T at(const T*, int64_t, int64_t) const`; AccessB too.
             * @param scratch Device memory for the packed panels; may be null.
             * @param c Must not overlap a, b or scratch.
             * @see modules/kernels/blas_seam_kernels/README.md
             */
            template<typename T, typename AccessA, typename AccessB,
                     typename Epilogue = GemmIdentityEpilogue>
            [[nodiscard]] sycl::event gemm_accessed(sycl::queue& queue,
                            int64_t m, int64_t n, int64_t k,
                            T alpha, const T* a, AccessA a_access, int64_t str_a,
                            const T* b, AccessB b_access, int64_t str_b,
                            T beta, T* c, OperandAccess c_access, int64_t str_c,
                            int64_t batch_size, void* scratch, std::size_t scratch_bytes,
                            const std::vector<sycl::event>& deps,
                            Epilogue epilogue = Epilogue{})
            {
                const PackedProductPlan plan =
                    plan_packed_product<T>(Device::profile_for(queue), m, n, k, batch_size);

                if (!packed_product_is_ready(plan, scratch, scratch_bytes))
                    return gemm_accessed<T, AccessA, AccessB, Epilogue>(
                        queue, m, n, k, alpha, a, a_access, str_a, b, b_access, str_b,
                        beta, c, c_access, str_c, batch_size, deps, epilogue);

                return Detail::packed_gemm<T>(queue, plan, m, n, k,
                                              alpha, a, a_access, str_a, b, b_access, str_b,
                                              beta, c, c_access, str_c, batch_size, scratch, deps,
                                              epilogue);
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
