/**************************************************************************/
/* positioned_random.hpp                                                  */
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
 * @file positioned_random.hpp
 * @brief Declares PositionedRandomOps, the random fills whose stream position is an operand.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>

namespace SushiBLAS
{
    /**
     * @class PositionedRandomOps
     * @brief Records random fills that read their stream position from a tensor each time they run.
     * @see modules/operations/random/README.md
     */
    class PositionedRandomOps final
    {
        public:
            /** @brief Builds a view over a recorder; it must outlive the view. */
            explicit PositionedRandomOps(TaskRecorder& recorder) noexcept
                : recorder_(recorder)
            {

            }

            PositionedRandomOps(const PositionedRandomOps&) = default;
            PositionedRandomOps& operator=(const PositionedRandomOps&) = delete;
            PositionedRandomOps(PositionedRandomOps&&) = default;
            PositionedRandomOps& operator=(PositionedRandomOps&&) = delete;
            ~PositionedRandomOps() = default;

            /**
             * @brief Fills a tensor from a Bernoulli distribution at the position @p position holds.
             * @param t FLOAT32 or FLOAT64, contiguous.
             * @param p Probability of a 1; within [0, 1].
             * @param position Rank-1 INT64 of K_RNG_POSITION_ELEMENTS elements, written by
             *        write_rng_position(); it must not overlap @p t.
             * @throws std::invalid_argument when p is outside [0, 1], t is HALF, or an operand breaks its rule.
             */
            TaskHandle bernoulli(Tensor& t, double p, const Tensor& position);

        private:
            /// @brief The recorder every fill is recorded through.
            TaskRecorder& recorder_;
    };
} // namespace SushiBLAS
