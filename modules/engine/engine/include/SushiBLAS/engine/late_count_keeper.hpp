/**************************************************************************/
/* late_count_keeper.hpp                                                  */
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
 * @file late_count_keeper.hpp
 * @brief Declares LateCount and LateCountKeeper, the port a TaskRecorder borrows a late count through and keeps its owner in.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    /** @brief Holds what one late-count task needs: the owner token and the words of the count. */
    struct LateCount
    {
        /// @brief The token the runtime watches; its deleter holds every storage of the task.
        SushiRuntime::LivenessToken owner;

        /// @brief The whole allocation of the count, borrowed as the runtime word buffer.
        SushiRuntime::Buffer<std::uint32_t> words;
    };

    /**
     * @brief Borrows the count of a late-count task and keeps the owner token of its storages.
     *
     * LateCountHold is the adapter over a runtime; HostLateCountKeeper is the adapter that makes
     * no runtime and counts the tokens it keeps.
     * @see modules/engine/engine/README.md
     */
    class LateCountKeeper
    {
        public:
            LateCountKeeper(const LateCountKeeper&) = delete;
            LateCountKeeper& operator=(const LateCountKeeper&) = delete;
            LateCountKeeper(LateCountKeeper&&) = delete;
            LateCountKeeper& operator=(LateCountKeeper&&) = delete;
            virtual ~LateCountKeeper() = default;

            /**
             * @brief Makes the owner token of a late task and borrows its count under it.
             * @return The token, which holds every storage of operands and of count, and the
             *         allocation of the count as words.
             * @throws std::runtime_error when the adapter refuses the count.
             * @pre count has storage.
             */
            [[nodiscard]] virtual LateCount borrow(const OperandSet& operands,
                                                   const Tensor& count) const = 0;

            /** @brief Keeps owner until the adapter owner lets the recording go. */
            virtual void keep(SushiRuntime::LivenessToken owner) = 0;

        protected:
            /** @brief Lets an adapter construct itself; the port is never built alone. */
            LateCountKeeper() = default;
    };
} // namespace SushiBLAS
