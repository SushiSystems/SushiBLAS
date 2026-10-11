/**************************************************************************/
/* fixed_order_scan.hpp                                                   */
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
 * @file fixed_order_scan.hpp
 * @brief Declares the live widths of a tiled level plan laid out for a capacity.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <SushiBLAS/graph_primitives/fixed_order_reduce.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        namespace Detail
        {
            /**
             * @brief Holds the live width of one level of a plan laid out for a capacity.
             * @see modules/foundation/graph_primitives/README.md
             */
            struct LiveTileLevel
            {
                std::size_t in_count = 0;  /**< Live values this level folds. */
                std::size_t out_count = 0; /**< Live partials it produces. */
            };

            /**
             * @brief Lays out the levels a K_REDUCE_TILE tree has for a capacity.
             * @return The levels in execution order; empty when capacity is zero.
             */
            [[nodiscard]] inline std::vector<ReduceLevel> plan_tiled_levels(std::size_t capacity)
            {
                return plan_reduction(capacity);
            }

            /**
             * @brief Returns how many values a tiled level produces from live_in values.
             * @return ceil(live_in / K_REDUCE_TILE); zero for zero.
             */
            [[nodiscard]] inline std::size_t next_live_tile_count(std::size_t live_in) noexcept
            {
                return (live_in == 0) ? 0 : (live_in + K_REDUCE_TILE - 1) / K_REDUCE_TILE;
            }

            /**
             * @brief Fills a caller-owned array with the per-level widths of a live count.
             * @param level_count Number of levels, from plan_tiled_levels(capacity).size().
             * @param live_n      Live element count; must not exceed that capacity.
             * @param out         Array of at least level_count entries.
             */
            inline void fill_live_tile_levels(std::size_t level_count, std::size_t live_n,
                                              LiveTileLevel* out) noexcept
            {
                std::size_t count = live_n;
                for (std::size_t i = 0; i < level_count; ++i)
                {
                    out[i].in_count = count;
                    out[i].out_count = next_live_tile_count(count);
                    count = out[i].out_count;
                }
            }

            /**
             * @brief Writes the per-level widths of a live count as words a kernel reads.
             * @param level_count Number of levels, from plan_tiled_levels(capacity).size().
             * @param live_n      Live element count; must not exceed that capacity.
             * @param out         2 * level_count words: the in_count of level i at i, its
             *                    out_count at level_count + i.
             */
            inline void fill_live_level_words(std::size_t level_count, std::size_t live_n,
                                              std::uint32_t* out) noexcept
            {
                std::size_t count = live_n;
                for (std::size_t i = 0; i < level_count; ++i)
                {
                    out[i] = static_cast<std::uint32_t>(count);
                    count = next_live_tile_count(count);
                    out[level_count + i] = static_cast<std::uint32_t>(count);
                }
            }

            /**
             * @brief Returns the per-level widths of a live count over the levels of a capacity.
             * @param live_n Live element count; must not exceed the capacity the levels were
             *        planned for.
             */
            [[nodiscard]] inline std::vector<LiveTileLevel> live_tile_levels(
                const std::vector<ReduceLevel>& capacity_levels, std::size_t live_n)
            {
                std::vector<LiveTileLevel> live(capacity_levels.size());
                fill_live_tile_levels(capacity_levels.size(), live_n, live.data());
                return live;
            }

        } // namespace Detail
    } // namespace GraphPrimitives
} // namespace SushiBLAS
