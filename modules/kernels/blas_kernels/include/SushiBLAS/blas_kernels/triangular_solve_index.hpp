/**************************************************************************/
/* triangular_solve_index.hpp                                             */
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

/**
 * @file triangular_solve_index.hpp
 * @brief Declares the block walk, trailing range and stream split of the blocked triangular solves, usable on the host and the device.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Indexing
        {
            /** @brief Returns whether op(A) is upper triangular, given the stored triangle and the transpose. */
            [[nodiscard]] constexpr bool effective_upper(bool transposed, bool stored_upper) noexcept
            {
                return transposed != stored_upper;
            }

            /** @brief Holds a range of steps: first up to, and not including, last. */
            struct StepRange
            {
                int64_t first = 0;  /**< The first step. */
                int64_t last = 0;   /**< One past the last step. */
            };

            /**
             * @brief Returns the steps of the diagonal block a blocked solve takes on pass `pass`.
             * @param blocks Blocks that cover `order` steps.
             * @param width Steps per block.
             * @param forward Whether a step needs the steps below it, so the blocks run top down.
             * @pre pass is below blocks.
             */
            [[nodiscard]] constexpr StepRange solve_block(int64_t pass, int64_t blocks,
                                                          int64_t width, int64_t order,
                                                          bool forward) noexcept
            {
                const int64_t position = forward ? pass : blocks - 1 - pass;
                const int64_t first = position * width;
                const int64_t end = first + width;
                return StepRange{first, end < order ? end : order};
            }

            /** @brief Holds the steps one solved block still updates: `count` steps from `first`. */
            struct TrailingRange
            {
                int64_t first = 0;  /**< The first step updated. */
                int64_t count = 0;  /**< Steps updated; zero or less when there are none. */
            };

            /**
             * @brief Returns the steps that lie past a solved block, nearest first, at most `reach` of them.
             * @param block The solved block.
             * @param order Steps of the whole substitution.
             * @param reach Later steps one solved step is subtracted from, at most.
             */
            [[nodiscard]] constexpr TrailingRange trailing_range(const StepRange& block, int64_t order,
                                                                 int64_t reach,
                                                                 bool forward) noexcept
            {
                const int64_t remaining = forward ? order - block.last : block.first;
                const int64_t count = remaining < reach ? remaining : reach;
                return TrailingRange{forward ? block.last : block.first - count, count};
            }

            /** @brief Returns the first column of a row inside the block: the later of the block start and the row start. */
            [[nodiscard]] constexpr int64_t block_first_column(int64_t block_first,
                                                               int64_t row_first) noexcept
            {
                return block_first > row_first ? block_first : row_first;
            }

            /** @brief Returns one past the last column of a row inside the block: the earlier of the block end and the row end. */
            [[nodiscard]] constexpr int64_t block_end_column(int64_t block_last,
                                                             int64_t row_end) noexcept
            {
                return block_last < row_end ? block_last : row_end;
            }

            /** @brief Holds the matrix of a batch and the right-hand-side vector one stream solves. */
            struct StreamId
            {
                int64_t batch = 0;  /**< Matrix of the batch. */
                int64_t vector = 0; /**< Right-hand-side vector of that matrix. */
            };

            /** @brief Splits a stream index into its batch entry and vector. */
            [[nodiscard]] constexpr StreamId split_stream(int64_t stream, int64_t vectors) noexcept
            {
                return StreamId{stream / vectors, stream % vectors};
            }

            /** @brief Holds the batch entry, step and vector of one unknown the trailing update touches. */
            struct UpdateUnknown
            {
                int64_t batch = 0;  /**< Matrix of the batch. */
                int64_t step = 0;   /**< Step of the substitution. */
                int64_t vector = 0; /**< Right-hand-side vector. */
            };

            /**
             * @brief Splits an element of the trailing update into batch, step and vector.
             * @param trailing The steps still to be solved.
             * @param step_is_innermost Whether consecutive steps of one vector are adjacent in B,
             *        so the step runs fastest.
             */
            [[nodiscard]] constexpr UpdateUnknown update_unknown(int64_t element,
                                                                 const TrailingRange& trailing,
                                                                 int64_t vectors,
                                                                 bool step_is_innermost) noexcept
            {
                const int64_t per_matrix = trailing.count * vectors;
                const int64_t within = element % per_matrix;
                return UpdateUnknown{
                    element / per_matrix,
                    trailing.first + (step_is_innermost ? within % trailing.count : within / vectors),
                    step_is_innermost ? within / trailing.count : within % vectors};
            }
        }
    }
}
