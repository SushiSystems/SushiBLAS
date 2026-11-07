/**************************************************************************/
/* scan_passes.hpp                                                        */
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
 * @file scan_passes.hpp
 * @brief Declares the host-callable arithmetic of the passes a prefix scan records.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <SushiBLAS/graph_primitives/fixed_order_fold.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        namespace Detail
        {
            /**
             * @brief Writes the scan of an empty input: a zero first offset and a zero total.
             * @param offsets Receives offsets[0].
             * @param total   Receives total[0].
             * @see modules/foundation/graph_primitives/README.md
             */
            inline void publish_empty_scan(std::uint32_t* offsets, std::uint32_t* total) noexcept
            {
                offsets[0] = 0;
                total[0] = 0;
            }

            /**
             * @brief Scans the whole top level from zero and publishes the sum.
             * @param src           Holds in_count values.
             * @param dst           Receives the exclusive prefix sums, modulo 2^32.
             * @param total         Receives the sum in total[0].
             * @param sentinel_base Receives the sum at sentinel_base[in_count]; null writes none.
             * @see modules/foundation/graph_primitives/README.md
             */
            inline void scan_top_pass(const std::uint32_t* src, std::uint32_t* dst,
                                      std::size_t in_count, std::uint32_t* total,
                                      std::uint32_t* sentinel_base) noexcept
            {
                std::uint32_t running = 0;
                for (std::size_t j = 0; j < in_count; ++j)
                {
                    dst[j] = running;
                    running += src[j];
                }
                total[0] = running;
                if (sentinel_base != nullptr)
                    sentinel_base[in_count] = running;
            }

            /**
             * @brief Scans one tile of a level from the prefix the level above wrote for it.
             * @param tile     Tile index; tile * K_REDUCE_TILE must be below in_count.
             * @param in_count Live values of the level; a short last tile stops there.
             * @param base     Exclusive prefix of the tile's first value.
             * @see modules/foundation/graph_primitives/README.md
             */
            inline void scan_tile(const std::uint32_t* src, std::uint32_t* dst, std::size_t tile,
                                  std::size_t in_count, std::uint32_t base) noexcept
            {
                const std::size_t begin = tile * K_REDUCE_TILE;
                std::size_t end = begin + K_REDUCE_TILE;
                if (end > in_count)
                    end = in_count;
                std::uint32_t running = base;
                for (std::size_t j = begin; j < end; ++j)
                {
                    dst[j] = running;
                    running += src[j];
                }
            }

            /**
             * @brief Copies the total into the offset the live count names.
             * @param live_n Live count; offsets[live_n] receives total[0].
             * @see modules/foundation/graph_primitives/README.md
             */
            inline void close_scan(std::uint32_t* offsets, const std::uint32_t* total,
                                   std::size_t live_n) noexcept
            {
                offsets[live_n] = total[0];
            }

        } // namespace Detail
    } // namespace GraphPrimitives
} // namespace SushiBLAS
