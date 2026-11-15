/**************************************************************************/
/* operand_pointers.hpp                                                   */
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
 * @file operand_pointers.hpp
 * @brief Declares OperandBinding and OperandPointers, how a recorded body finds the address of an
 *        operand its task bound by handle.
 * @author Mustafa Garip
 */

#pragma once

#include <limits>
#include <cstddef>

namespace SushiBLAS
{
    /**
     * @brief Names one bound operand: its slot in the table of the task and the bytes of its first
     *        element past the base of its allocation.
     * @see modules/engine/engine/README.md
     */
    struct OperandBinding
    {
        /// @brief The slot of a tensor with no storage, which has no address.
        static constexpr std::size_t K_ABSENT = std::numeric_limits<std::size_t>::max();

        std::size_t slot = K_ABSENT; /**< Position among the operands the task bound, in bind order; K_ABSENT for none. */
        std::size_t byte_offset = 0; /**< Bytes from the base of the allocation to the first element. */
    };

    /**
     * @brief Views the base address of each allocation a task bound, one per slot.
     *
     * A graph sink fills it from the pointers the runtime bound on the device the node runs on; a
     * host sink fills it from the bases the tasks recorded. The view is valid during one run.
     * @see modules/engine/engine/README.md
     */
    class OperandPointers final
    {
        public:
            /** @brief Views no pointer. */
            OperandPointers() noexcept = default;

            /** @brief Views @p count base addresses starting at @p table. */
            OperandPointers(void* const* table, std::size_t count) noexcept
                : table_(table), count_(count)
            {

            }

            /** @brief Returns the number of bound allocations. */
            [[nodiscard]] std::size_t size() const noexcept { return count_; }

            /** @brief Tells whether the view holds no pointer. */
            [[nodiscard]] bool is_empty() const noexcept { return count_ == 0; }

            /**
             * @brief Returns the address of the first element of the operand, or null for an
             *        operand with no storage.
             * @pre operand.slot is below size() or OperandBinding::K_ABSENT.
             */
            [[nodiscard]] void* address(const OperandBinding& operand) const noexcept
            {
                if (operand.slot == OperandBinding::K_ABSENT)
                    return nullptr;
                return static_cast<char*>(table_[operand.slot]) + operand.byte_offset;
            }

            /**
             * @brief Returns the address of the first element of the operand as a T*.
             * @pre operand.slot is below size() or OperandBinding::K_ABSENT.
             */
            template <class T>
            [[nodiscard]] T* as(const OperandBinding& operand) const noexcept
            {
                return static_cast<T*>(address(operand));
            }

        private:
            /// @brief The base address of each bound allocation.
            void* const* table_ = nullptr;

            /// @brief Number of addresses in table_.
            std::size_t count_ = 0;
    };
} // namespace SushiBLAS
