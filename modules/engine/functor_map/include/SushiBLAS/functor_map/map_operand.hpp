/**************************************************************************/
/* map_operand.hpp                                                        */
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
 * @file map_operand.hpp
 * @brief Declares MapOperand and the calls that bind a tensor to a task or record its address.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <SushiRuntime/handle/bound_operands.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/operand_binding.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/operand_set.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        namespace Detail
        {
            /**
             * @brief Names where a map task finds one operand at each run: in the table its task
             *        bound, or at the address recorded with it.
             *
             * An operand with no storage has neither and reads as null.
             * @see modules/engine/functor_map/README.md
             */
            struct MapOperand
            {
                /** @brief Names an operand with no storage. */
                MapOperand() = default;

                /** @brief Names an operand the task bound; it records no address. */
                MapOperand(const OperandBinding& bound) noexcept : binding(bound)
                {

                }

                /** @brief Names an operand by the address recorded with the task. */
                MapOperand(const void* address) noexcept : recorded(address)
                {

                }

                /** @brief Returns the address of the first element for the run whose operands are @p operands. */
                [[nodiscard]] const void* resolve(const SushiRuntime::BoundOperands& operands) const
                {
                    return binding.slot == OperandBinding::K_ABSENT ? recorded
                                                                    : operand_address<const std::byte>(operands, binding);
                }

                /// @brief Where the task's bound table holds the operand; absent for a recorded address.
                OperandBinding binding;

                /// @brief The address recorded with the task; null for a bound operand.
                const void* recorded = nullptr;
            };

            /**
             * @brief Checks the storage of an operand of a map-shaped task.
             * @tparam Bind True when the task binds its operands, which a planned tensor may be;
             *         false when it records addresses, which a planned tensor has none of.
             * @throws std::invalid_argument as require_bindable_storage or require_storage states.
             */
            template<bool Bind>
            inline void require_map_storage(const Tensor& t, const char* op, const char* operand)
            {
                if constexpr (Bind)
                    require_bindable_storage(t, op, operand);
                else
                    require_storage(t, op, operand);
            }

            /**
             * @brief Returns where a task finds the first element of t at each run.
             * @tparam Bind True binds t on operands and records no address; false records the
             *         address of the first element and binds nothing.
             * @pre t has passed require_map_storage<Bind>.
             */
            template<bool Bind>
            [[nodiscard]] inline MapOperand map_operand(OperandSet& operands, const Tensor& t)
            {
                if constexpr (Bind)
                    return MapOperand(operands.bind(t));
                else
                    return MapOperand(t.data_or_null());
            }
        } // namespace Detail
    } // namespace Dispatch
} // namespace SushiBLAS
