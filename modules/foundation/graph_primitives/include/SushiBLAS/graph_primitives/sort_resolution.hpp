/**************************************************************************/
/* sort_resolution.hpp                                                    */
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
 * @file sort_resolution.hpp
 * @brief Declares the host-callable decisions a counting sort makes from its live sizes.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <SushiBLAS/graph_primitives/fixed_order_scan.hpp>
#include <SushiBLAS/graph_primitives/radix_plan.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        namespace Detail
        {
            /**
             * @brief Publishes the live element count and fills the digit plan of one run.
             * @param live_key_range Live key range; must not exceed the planned capacity.
             * @param n_state        Receives the live count at 0 and its live tile count at 1.
             * @param total          Receives the live count in total[0].
             * @param plan           Receives the digit plan; null when no element ever arrives.
             * @see modules/foundation/graph_primitives/README.md
             */
            inline void publish_sort_state(std::size_t digit_count, std::size_t live_n,
                                           std::size_t live_key_range, std::uint32_t* n_state,
                                           std::uint32_t* total, std::uint32_t* plan) noexcept
            {
                const std::size_t tiles = next_live_tile_count(live_n);
                n_state[0] = static_cast<std::uint32_t>(live_n);
                n_state[1] = static_cast<std::uint32_t>(tiles);
                total[0] = static_cast<std::uint32_t>(live_n);
                if (plan != nullptr)
                    fill_radix_plan(digit_count, live_n, tiles, live_key_range, plan);
            }

            /**
             * @brief Zeroes bucket k of a sort that never receives an element.
             * @param k            Bucket; key_capacity names the closing offset.
             * @param key_capacity Buckets the sort was planned for.
             * @param offsets      Receives offsets[k].
             * @param counts       Receives counts[k] for k below key_capacity; null skips counts.
             * @see modules/foundation/graph_primitives/README.md
             */
            inline void clear_empty_bucket(std::size_t k, std::size_t key_capacity,
                                           std::uint32_t* offsets, std::uint32_t* counts) noexcept
            {
                offsets[k] = 0;
                if (counts != nullptr && k < key_capacity)
                    counts[k] = 0;
            }

            /**
             * @brief Returns where bucket k starts: how many sorted keys are below k.
             * @param keys    The unsorted keys.
             * @param indices The sorted order, as indices into keys; live_n entries are live.
             * @see modules/foundation/graph_primitives/README.md
             */
            [[nodiscard]] inline std::uint32_t bucket_start(const std::uint32_t* keys,
                                                            const std::uint32_t* indices,
                                                            std::size_t live_n,
                                                            std::size_t k) noexcept
            {
                std::size_t low = 0;
                std::size_t high = live_n;
                while (low < high)
                {
                    const std::size_t middle = low + (high - low) / 2;
                    if (keys[indices[middle]] < k)
                        low = middle + 1;
                    else
                        high = middle;
                }
                return static_cast<std::uint32_t>(low);
            }

        } // namespace Detail
    } // namespace GraphPrimitives
} // namespace SushiBLAS
