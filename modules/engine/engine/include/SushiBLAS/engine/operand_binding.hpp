/**************************************************************************/
/* operand_binding.hpp                                                    */
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
 * @file operand_binding.hpp
 * @brief Declares OperandBinding and operand_address(), how a recorded body finds the address of an
 *        operand its task bound by handle in the runtime's view of the bound operands.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

#include <SushiRuntime/handle/bound_operands.hpp>

#include <SushiBLAS/core/logger.hpp>

namespace SushiBLAS
{
    /**
     * @brief Names one bound operand: its slot in the operands the task bound and the bytes of its
     *        first element past the base of its allocation.
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
     * @brief Returns the address of the first element of the operand as a T*, read from the
     *        runtime's view of the operands the task bound.
     *
     * Every storage block is raw bytes, so the view accepts any T at an offset aligned for it.
     * @return Null for an operand with no storage.
     * @throws std::runtime_error when the view refuses the slot, the element type or the offset.
     */
    template <class T>
    [[nodiscard]] T* operand_address(const SushiRuntime::BoundOperands& operands,
                                     const OperandBinding& operand)
    {
        if (operand.slot == OperandBinding::K_ABSENT)
            return nullptr;
        const SushiRuntime::OperandAddress<T> found = operands.at<T>(operand.slot, operand.byte_offset);
        SB_THROW_IF(!found, "operand slot {} cannot be read as the element type asked for: status {}.",
                    operand.slot, static_cast<int>(found.status));
        return found.address;
    }
} // namespace SushiBLAS
