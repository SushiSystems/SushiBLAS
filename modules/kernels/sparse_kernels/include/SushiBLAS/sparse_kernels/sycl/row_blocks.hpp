/**************************************************************************/
/* row_blocks.hpp                                                         */
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
 * @file row_blocks.hpp
 * @brief Defines the row-block scatter that the transposed and the symmetric product share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/row_block_index.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Holds the ceiling on the row blocks of one scattering product.
             * @see kernels/README.md
             */
            inline constexpr std::size_t K_MAX_SCATTER_BLOCKS = Indexing::K_MAX_SCATTER_BLOCKS;

            /**
             * @brief Holds how many partial sums a scattering product may keep per stored entry.
             * @see kernels/README.md
             */
            inline constexpr std::size_t K_SCATTER_PARTIALS_PER_ENTRY =
                Indexing::K_SCATTER_PARTIALS_PER_ENTRY;

            /**
             * @brief Returns how many row blocks a scattering product over nnz entries uses.
             * @param outputs Elements of the result.
             * @return At least 1, or 0 when outputs is 0; each block needs outputs partial sums.
             * @see kernels/README.md
             */
            [[nodiscard]] inline std::size_t scatter_blocks(const Device::Profile& profile,
                                                            int64_t outputs, int64_t nnz) noexcept
            {
                return Indexing::scatter_block_count(
                    Indexing::resident_blocks(profile.is_cpu, profile.compute_units,
                                              Device::K_WAVES_PER_COMPUTE_UNIT),
                    outputs, nnz);
            }

            namespace Detail
            {
                /**
                 * @brief Stores output j of y = alpha * (the sum of every block's partial)
                 *        + beta * y.
                 * @param partials blocks slices of outputs elements; block b alone wrote slice b.
                 * @param j Output in [0, outputs).
                 * @param beta Exactly zero skips reading y.
                 * @see kernels/README.md
                 */
                template<typename T>
                inline void fold_block_output(const Accumulator<T>* partials, int64_t outputs,
                                              int64_t blocks, int64_t j, T alpha, T beta, T* y,
                                              int64_t incy)
                {
                    using W = Accumulator<T>;

                    // Folding the blocks in ascending order keeps ascending row order.
                    W sum(0);
                    for (int64_t block = 0; block < blocks; ++block)
                        sum += partials[Indexing::partial_index(block, outputs, j)];

                    const W scaled = multiply(W(alpha), sum);
                    const int64_t out = j * incy;
                    y[out] = (beta == T(0))
                                 ? static_cast<T>(scaled)
                                 : static_cast<T>(scaled + multiply(W(beta), W(y[out])));
                }

                /**
                 * @brief Stores y = alpha * (the sum of every block's partial) + beta * y, one
                 *        work-item per output, behind the event of the scatter.
                 * @param partials blocks slices of outputs elements; block b alone wrote slice b.
                 * @param beta Exactly zero skips reading y.
                 * @see kernels/README.md
                 */
                template<typename T>
                [[nodiscard]] sycl::event fold_blocks(sycl::queue& queue,
                                                      const sycl::event& scattered,
                                                      int64_t outputs, int64_t blocks,
                                                      const Accumulator<T>* partials, T alpha,
                                                      T beta, T* y, int64_t incy)
                {
                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(scattered);
                        Launch::indexed(h, queue, static_cast<std::size_t>(outputs),
                                        [=](std::size_t index)
                        {
                            fold_block_output(partials, outputs, blocks,
                                              static_cast<int64_t>(index), alpha, beta, y, incy);
                        });
                    });
                }
            } // namespace Detail
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
