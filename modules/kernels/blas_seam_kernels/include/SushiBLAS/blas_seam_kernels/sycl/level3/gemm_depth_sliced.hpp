/**************************************************************************/
/* gemm_depth_sliced.hpp                                                  */
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
/*                                                                        */

/**
 * @file gemm_depth_sliced.hpp
 * @brief Declares the depth-sliced GEMM, which sums slices of K apart and folds them in order.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/kernel_variant.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_epilogue.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_catalog.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_depth_slice_plan.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm_store.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Writes each element of C from the partial sums of its slices, added in
                 *        slice order.
                 * @param partials Row-major slices; slice s of entry i starts at
                 *        partials + (i * slices + s) * m * n.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename T, typename Epilogue>
                [[nodiscard]] sycl::event fold_depth_slices(
                    sycl::queue& queue, int64_t slices, int64_t m, int64_t n, const T* partials,
                    T alpha, T beta, T* c, OperandAccess c_access, int64_t str_c,
                    int64_t batch_size, const std::vector<sycl::event>& deps, Epilogue epilogue)
                {
                    const int64_t matrix = m * n;
                    const GemmStore<T, Epilogue> store(alpha, beta, c_access, epilogue);

                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        h.parallel_for(
                            sycl::range<1>(static_cast<std::size_t>(batch_size * matrix)),
                            [=](sycl::id<1> id)
                        {
                            const int64_t flat = static_cast<int64_t>(id[0]);
                            const int64_t batch_id = flat / matrix;
                            const int64_t element = flat % matrix;
                            const T* first = partials + batch_id * slices * matrix + element;

                            // Slices are added in the same order on every run and every device.
                            Accumulator<T> sum = Accumulator<T>(first[0]);
                            for (int64_t s = 1; s < slices; ++s)
                                sum = sum + Accumulator<T>(first[s * matrix]);

                            store(c + batch_id * str_c, element / n, element % n, sum);
                        });
                    });
                }

                /**
                 * @brief Computes GEMM as slices of K summed by the Balanced tiled kernel into
                 *        scratch, then folded into C.
                 * @pre depth_slices_are_ready(plan, scratch, the bytes behind scratch) holds.
                 * @param c Must not overlap a, b or scratch.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename T, typename AccessA, typename AccessB, typename Epilogue>
                [[nodiscard]] sycl::event gemm_depth_sliced(
                    sycl::queue& queue, const GemmDepthSlicePlan& plan,
                    int64_t m, int64_t n, int64_t k,
                    T alpha, const T* a, AccessA a_access, int64_t str_a,
                    const T* b, AccessB b_access, int64_t str_b,
                    T beta, T* c, OperandAccess c_access, int64_t str_c,
                    int64_t batch_size, void* scratch,
                    const std::vector<sycl::event>& deps, Epilogue epilogue)
                {
                    using Balanced = GemmTile<T, GemmTileRole::Balanced>;

                    Device::note_kernel_variant(Device::KernelVariant::GemmDepthSliced);

                    T* partials = static_cast<T*>(scratch);
                    const int64_t matrix = m * n;

                    const sycl::event summed = gemm_local_tiled<T, Balanced>(
                        queue, m, n, k,
                        T(1), a, a_access, str_a,
                        b, b_access, str_b,
                        T(0), partials, OperandAccess{n, 1, false}, plan.slices * matrix,
                        batch_size, deps, GemmIdentityEpilogue{},
                        DepthSlices{plan.slices, plan.depth, matrix});

                    return fold_depth_slices<T>(queue, plan.slices, m, n, partials,
                                                alpha, beta, c, c_access, str_c,
                                                batch_size, {summed}, epilogue);
                }

                /**
                 * @brief Computes GEMM depth-sliced when the profile asks for slices and scratch
                 *        holds them, else as gemm_accessed without scratch does.
                 * @param scratch Device memory for the partial sums; may be null.
                 * @param c Must not overlap a, b or scratch.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename T, typename AccessA, typename AccessB, typename Epilogue>
                [[nodiscard]] sycl::event gemm_sliced_or_accessed(
                    sycl::queue& queue,
                    int64_t m, int64_t n, int64_t k,
                    T alpha, const T* a, AccessA a_access, int64_t str_a,
                    const T* b, AccessB b_access, int64_t str_b,
                    T beta, T* c, OperandAccess c_access, int64_t str_c,
                    int64_t batch_size, void* scratch, std::size_t scratch_bytes,
                    const std::vector<sycl::event>& deps, Epilogue epilogue)
                {
                    const GemmDepthSlicePlan plan = plan_gemm_depth_slices<T>(
                        Device::profile_for(queue), m, n, k, batch_size);

                    if (!depth_slices_are_ready(plan, scratch, scratch_bytes))
                        return gemm_accessed<T, AccessA, AccessB, Epilogue>(
                            queue, m, n, k, alpha, a, a_access, str_a, b, b_access, str_b,
                            beta, c, c_access, str_c, batch_size, deps, epilogue);

                    return gemm_depth_sliced<T>(queue, plan, m, n, k,
                                                alpha, a, a_access, str_a, b, b_access, str_b,
                                                beta, c, c_access, str_c, batch_size, scratch,
                                                deps, epilogue);
                }
            } // namespace Detail
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
