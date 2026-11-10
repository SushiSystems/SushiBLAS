/**************************************************************************/
/* host_late_count_keeper.hpp                                             */
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
 * @file host_late_count_keeper.hpp
 * @brief Declares HostLateCountKeeper, the LateCountKeeper that makes no runtime and counts the owners it keeps.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <SushiBLAS/engine/late_count_keeper.hpp>

namespace SushiBLAS
{
    /**
     * @brief Hands out a late count with no words and counts the owner tokens it is given, so a
     *        TaskRecorder over a host sink records late-count tasks in a test that makes no runtime.
     * @see modules/engine/engine/README.md
     */
    class HostLateCountKeeper final : public LateCountKeeper
    {
        public:
            /** @brief Keeps nothing. */
            HostLateCountKeeper() noexcept = default;

            /** @brief Returns a late count whose token and words are empty. */
            [[nodiscard]] LateCount borrow(const OperandSet&, const Tensor&) const override
            {
                return LateCount{};
            }

            /** @brief Counts the owner and drops it. */
            void keep(SushiRuntime::LivenessToken) override { ++kept_; }

            /** @brief Returns how many owners were kept. */
            [[nodiscard]] std::size_t kept_count() const noexcept { return kept_; }

        private:
            /// @brief How many owners were kept.
            std::size_t kept_ = 0;
    };
} // namespace SushiBLAS
