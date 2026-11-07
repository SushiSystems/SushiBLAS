/**************************************************************************/
/* row_block_index.hpp                                                    */
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
 * @file row_block_index.hpp
 * @brief Declares the row-block geometry of the scattering sparse products, shared by the host and the device.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Indexing
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
             * @brief Returns how many blocks a device keeps resident: its compute units on a CPU,
             *        and its compute units times the waves per unit otherwise.
             */
            [[nodiscard]] constexpr std::size_t resident_blocks(bool is_cpu,
                                                                std::size_t compute_units,
                                                                std::size_t waves_per_unit) noexcept
            {
                return is_cpu ? compute_units : compute_units * waves_per_unit;
            }

            /**
             * @brief Returns how many row blocks a scattering product over nnz entries uses.
             * @param resident The blocks the device keeps resident.
             * @param outputs Elements of the result.
             * @return At least 1, or 0 when outputs is 0; each block needs outputs partial sums.
             * @see kernels/README.md
             */
            [[nodiscard]] constexpr std::size_t scatter_block_count(std::size_t resident,
                                                                    int64_t outputs,
                                                                    int64_t nnz) noexcept
            {
                if (outputs <= 0)
                    return 0;

                const std::size_t affordable =
                    (K_SCATTER_PARTIALS_PER_ENTRY * static_cast<std::size_t>(nnz)) /
                    static_cast<std::size_t>(outputs);

                const std::size_t blocks = std::min({resident, affordable, K_MAX_SCATTER_BLOCKS});
                return blocks > 0 ? blocks : std::size_t{1};
            }

            /**
             * @brief Returns the entry at which one of the blocks that split nnz entries evenly
             *        starts: `nnz * block / blocks`, rounded down.
             * @param block In [0, blocks].
             * @param blocks Must be positive.
             */
            [[nodiscard]] constexpr int64_t block_target(int64_t nnz, int64_t block,
                                                         int64_t blocks) noexcept
            {
                return (nnz / blocks) * block + ((nnz % blocks) * block) / blocks;
            }

            /**
             * @brief Returns the first row of one of the blocks that split the stored entries
             *        evenly.
             * @param block In [0, blocks]; blocks itself gives the end of the last block.
             * @return The first row whose entries start at or after entry nnz * block / blocks.
             * @see kernels/README.md
             */
            template<typename Index>
            [[nodiscard]] constexpr int64_t block_first_row(const CsrPattern<Index>& pattern,
                                                            int64_t block, int64_t blocks) noexcept
            {
                const int64_t target = block_target(pattern.nnz, block, blocks);

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
            [[nodiscard]] constexpr int64_t block_end_row(const CsrPattern<Index>& pattern,
                                                          int64_t block, int64_t blocks) noexcept
            {
                return (block + 1 == blocks) ? pattern.rows
                                             : block_first_row(pattern, block + 1, blocks);
            }

            /** @brief Returns where output j of block `block` sits in a buffer of `outputs`-wide slices. */
            [[nodiscard]] constexpr int64_t partial_index(int64_t block, int64_t outputs,
                                                          int64_t j) noexcept
            {
                return block * outputs + j;
            }
        }
    }
}
