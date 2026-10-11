/**************************************************************************/
/* fixed_order_reduce.hpp                                                 */
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
 * @file fixed_order_reduce.hpp
 * @brief Plans the levels of a fixed-tile tree reduction.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <vector>

#include <SushiBLAS/graph_primitives/fixed_order_fold.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        namespace Detail
        {
            /**
             * @brief Describes one level of a fixed-tile tree reduction.
             * @see modules/foundation/graph_primitives/README.md
             */
            struct ReduceLevel
            {
                std::size_t in_count = 0;   /**< Values this level folds. */
                std::size_t out_count = 0;  /**< Partials it produces. */
                std::size_t in_offset = 0;  /**< Element offset of its input in scratch. */
                std::size_t out_offset = 0; /**< Element offset of its output in scratch. */
            };

            /**
             * @brief Lays out the levels of a fixed-tile reduction over n values.
             * @return The levels in execution order; empty when n is zero.
             * @see modules/foundation/graph_primitives/README.md
             */
            [[nodiscard]] inline std::vector<ReduceLevel> plan_reduction(std::size_t n)
            {
                std::vector<ReduceLevel> levels;
                if (n == 0)
                    return levels;

                std::size_t count = n;
                std::size_t scratch_used = 0;
                std::size_t previous_out_offset = 0;

                while (true)
                {
                    ReduceLevel level;
                    level.in_count = count;
                    level.out_count = (count + K_REDUCE_TILE - 1) / K_REDUCE_TILE;
                    level.in_offset = previous_out_offset;
                    level.out_offset = scratch_used;

                    scratch_used += level.out_count;
                    previous_out_offset = level.out_offset;
                    count = level.out_count;

                    levels.push_back(level);
                    if (count == 1)
                        break;
                }
                return levels;
            }

            /**
             * @brief Returns the scratch element count a planned reduction needs.
             * @return Elements to allocate; zero when the reduction is one level.
             */
            [[nodiscard]] inline std::size_t reduce_scratch_count(
                const std::vector<ReduceLevel>& levels) noexcept
            {
                std::size_t total = 0;
                for (std::size_t i = 0; i + 1 < levels.size(); ++i)
                    total += levels[i].out_count;
                return total;
            }

        } // namespace Detail
    } // namespace GraphPrimitives
} // namespace SushiBLAS
