/**************************************************************************/
/* task_cost.hpp                                                          */
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
 * @file task_cost.hpp
 * @brief Declares the value functions that turn an operation, a shape and an operand set into the
 *        access and the cost of one recorded node.
 * @author Mustafa Garip
 */

#pragma once

#include <optional>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/cost/cost.hpp>
#include <SushiBLAS/cost/cost_shape.hpp>
#include <SushiBLAS/cost/cost_table.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        namespace Detail
        {
            /**
             * @brief Returns the runtime's name for the ceiling class of a real dtype.
             * @return F16 or F32; Unstated for a type the runtime names no class for.
             */
            [[nodiscard]] constexpr SushiRuntime::NodeDType
            node_dtype_of(Core::DataType dtype) noexcept
            {
                switch (dtype)
                {
                    case Core::DataType::Half:
                        return SushiRuntime::NodeDType::F16;
                    case Core::DataType::Float32:
                        return SushiRuntime::NodeDType::F32;
                    default:
                        return SushiRuntime::NodeDType::Unstated;
                }
            }
        } // namespace Detail

        /**
         * @brief Builds the access of operands: its regions, and the owners of borrowed storage a
         *        run checks first.
         * @see modules/engine/engine/README.md
         */
        [[nodiscard]] inline SushiRuntime::Access task_access(const OperandSet& operands)
        {
            SushiRuntime::Access access(operands.read_regions(), operands.write_regions());
            access.lives = operands.owner_liveness();
            return access;
        }

        /**
         * @brief Returns the cost of a task: the table's cost for the operation and shape, with
         *        the bytes the operands declare.
         * @see modules/engine/engine/README.md
         */
        [[nodiscard]] inline Cost task_cost(SushiRuntime::OperationID operation,
                                            const CostShape& shape, const OperandSet& operands)
        {
            return cost_of(operation, shape).with_bytes(operands.declared_bytes());
        }

        /**
         * @brief Returns the cost a runtime node states.
         * @return Empty when cost is uncosted.
         * @see modules/engine/engine/README.md
         */
        [[nodiscard]] inline std::optional<SushiRuntime::NodeCost>
        node_cost_of(const Cost& cost) noexcept
        {
            if (!cost.is_costed())
                return std::nullopt;

            SushiRuntime::NodeCost node_cost;
            node_cost.flops = cost.flops;
            node_cost.bytes = cost.bytes;
            node_cost.dtype = Detail::node_dtype_of(cost.dtype);
            return node_cost;
        }
    } // namespace Internal
} // namespace SushiBLAS
