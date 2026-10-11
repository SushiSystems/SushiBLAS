/**************************************************************************/
/* fixed_order_fold.hpp                                                   */
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
 * @file fixed_order_fold.hpp
 * @brief Defines the tile size of the fixed fold order and the host folds that follow it.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        /**
         * @brief Holds how many elements one work-item folds in sequence at every level.
         * @see modules/foundation/graph_primitives/README.md
         */
        inline constexpr std::size_t K_REDUCE_TILE = 256;

        /** @brief Holds the tree levels a streamed fold keeps, enough for any 64-bit count. */
        inline constexpr std::size_t K_STREAMED_FOLD_LEVELS = 8;

        /**
         * @brief Folds src[begin, end) left to right, seeded with src[begin].
         * @tparam Combine Binary functor called as op(accumulator, element).
         * @pre begin < end.
         */
        template <typename T, class Combine>
        [[nodiscard]] inline T fold_tile(const Combine& op, const T* src, std::size_t begin,
                                         std::size_t end)
        {
            T accumulator = src[begin];
            for (std::size_t j = begin + 1; j < end; ++j)
                accumulator = op(accumulator, src[j]);
            return accumulator;
        }

        /**
         * @brief Folds tile t of the first in_count values of src.
         * @tparam Combine Binary functor called as op(accumulator, element).
         * @pre t * K_REDUCE_TILE < in_count.
         */
        template <typename T, class Combine>
        [[nodiscard]] inline T fold_indexed_tile(const Combine& op, const T* src, std::size_t t,
                                                 std::size_t in_count)
        {
            const std::size_t begin = t * K_REDUCE_TILE;
            std::size_t end = begin + K_REDUCE_TILE;
            if (end > in_count)
                end = in_count;
            return fold_tile(op, src, begin, end);
        }

        /**
         * @brief Folds count values on the host in the fixed tile order, level by level.
         * @param values The values to fold; may be null only when count is zero.
         * @param identity The result when count is zero; it never enters the arithmetic otherwise.
         * @see modules/foundation/graph_primitives/README.md
         */
        template <typename T, class Combine>
        [[nodiscard]] inline T fold_fixed_order(const Combine& op, const T* values,
                                                std::size_t count, T identity)
        {
            if (count == 0)
                return identity;
            if (count <= K_REDUCE_TILE)
                return fold_tile(op, values, 0, count);

            std::vector<T> partials((count + K_REDUCE_TILE - 1) / K_REDUCE_TILE);
            const T* in = values;
            std::size_t in_count = count;
            while (in_count > 1)
            {
                const std::size_t out_count = (in_count + K_REDUCE_TILE - 1) / K_REDUCE_TILE;
                // Tile t writes index t and reads from t * K_REDUCE_TILE up, so no write passes a read.
                for (std::size_t tile = 0; tile < out_count; ++tile)
                    partials[tile] = fold_indexed_tile(op, in, tile, in_count);
                in = partials.data();
                in_count = out_count;
            }
            return partials[0];
        }

        /**
         * @brief Folds count elements in the fixed tile order on the stack, reading each once.
         * @param read Returns element i for i in [0, count).
         * @param identity The result when count is zero; it never enters the arithmetic otherwise.
         * @see modules/foundation/graph_primitives/README.md
         */
        template <typename T, class Read, class Combine>
        [[nodiscard]] inline T fold_fixed_order_streamed(const Combine& op, std::size_t count,
                                                         T identity, Read read) noexcept
        {
            if (count == 0)
                return identity;
            std::size_t top = 0;
            for (std::uint64_t level_count = count; level_count > 1; ++top)
                level_count = level_count / K_REDUCE_TILE + (level_count % K_REDUCE_TILE != 0);
            if (top == 0)
                return read(std::size_t{0});
            top -= 1;

            T partial[K_STREAMED_FOLD_LEVELS] = {identity, identity, identity, identity,
                                                 identity, identity, identity, identity};
            std::size_t filled[K_STREAMED_FOLD_LEVELS] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (std::uint64_t index = 0; index < count; ++index)
            {
                T carry = read(static_cast<std::size_t>(index));
                for (std::size_t level = 0; level <= top; ++level)
                {
                    partial[level] = filled[level] == 0 ? carry : op(partial[level], carry);
                    filled[level] += 1;
                    if (level == top || filled[level] != K_REDUCE_TILE)
                        break;
                    carry = partial[level];
                    filled[level] = 0;
                }
            }
            for (std::size_t level = 0; level < top; ++level)
            {
                if (filled[level] == 0)
                    continue;
                const T carry = partial[level];
                filled[level] = 0;
                partial[level + 1] =
                    filled[level + 1] == 0 ? carry : op(partial[level + 1], carry);
                filled[level + 1] += 1;
            }
            return partial[top];
        }

    } // namespace GraphPrimitives
} // namespace SushiBLAS
