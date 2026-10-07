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
 * @file row_blocks.hpp
 * @brief Defines the row-block scatter that the transposed and the symmetric product share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <SushiBLAS/support/launch_1d.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include "kernels/sparse/csr_pattern.hpp"

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
            inline constexpr std::size_t K_MAX_SCATTER_BLOCKS = 256;

            /**
             * @brief Holds how many partial sums a scattering product may keep per stored entry.
             * @see kernels/README.md
             */
            inline constexpr std::size_t K_SCATTER_PARTIALS_PER_ENTRY = 2;

            /**
             * @brief Returns how many row blocks a scattering product over nnz entries uses.
             * @param outputs Elements of the result.
             * @return At least 1, or 0 when outputs is 0; each block needs outputs partial sums.
             * @see kernels/README.md
             */
            [[nodiscard]] inline std::size_t scatter_blocks(const Device::Profile& profile,
                                                            int64_t outputs, int64_t nnz) noexcept
            {
                if (outputs <= 0)
                    return 0;

                const std::size_t resident =
                    profile.is_cpu ? profile.compute_units
                                   : profile.compute_units * Device::K_WAVES_PER_COMPUTE_UNIT;
                const std::size_t affordable =
                    (K_SCATTER_PARTIALS_PER_ENTRY * static_cast<std::size_t>(nnz)) /
                    static_cast<std::size_t>(outputs);

                const std::size_t blocks = std::min({resident, affordable, K_MAX_SCATTER_BLOCKS});
                return blocks > 0 ? blocks : std::size_t{1};
            }

            namespace Detail
            {
                /**
                 * @brief Returns the first row of one of the blocks that split the stored entries
                 *        evenly.
                 * @param block In [0, blocks]; blocks itself gives the end of the last block.
                 * @return The first row whose entries start at or after entry nnz * block / blocks.
                 * @see kernels/README.md
                 */
                template<typename Index>
                [[nodiscard]] inline int64_t block_first_row(const CsrPattern<Index>& pattern,
                                                             int64_t block, int64_t blocks) noexcept
                {
                    const int64_t target = (pattern.nnz / blocks) * block +
                                           ((pattern.nnz % blocks) * block) / blocks;

                    int64_t low = 0;
                    int64_t high = pattern.rows;
                    while (low < high)
                    {
                        const int64_t middle = low + (high - low) / 2;
                        if (pattern.row_begin(middle) < target)
                            low = middle + 1;
                        else
                            high = middle;
                    }
                    return low;
                }

                /**
                 * @brief Returns one past the last row of a block: the next block's first row, or
                 *        the row count for the last block.
                 */
                template<typename Index>
                [[nodiscard]] inline int64_t block_end_row(const CsrPattern<Index>& pattern,
                                                           int64_t block, int64_t blocks) noexcept
                {
                    return (block + 1 == blocks) ? pattern.rows
                                                 : block_first_row(pattern, block + 1, blocks);
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
                    using W = Accumulator<T>;

                    const bool keeps_y = !(beta == T(0));

                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(scattered);
                        Launch::indexed(h, queue, static_cast<std::size_t>(outputs),
                                        [=](std::size_t index)
                        {
                            const int64_t j = static_cast<int64_t>(index);

                            // Folding the blocks in ascending order keeps ascending row order.
                            W sum(0);
                            for (int64_t block = 0; block < blocks; ++block)
                                sum += partials[block * outputs + j];

                            const W scaled = multiply(W(alpha), sum);
                            const int64_t out = j * incy;
                            y[out] = keeps_y ? static_cast<T>(scaled + multiply(W(beta), W(y[out])))
                                             : static_cast<T>(scaled);
                        });
                    });
                }
            } // namespace Detail
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
