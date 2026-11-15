/**************************************************************************/
/* rng_position_tensor.cpp                                                */
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
 * @file rng_position_tensor.cpp
 * @brief Defines the checks and the host write of a random stream position tensor.
 * @author Mustafa Garip
 */

#include <SushiBLAS/random/rng_position_tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        void require_rng_position(const Tensor& position, const char* op, const char* operand)
        {
            require_dtype(position, Core::DataType::Int64, op, operand);
            require_rank(position, 1, op, operand);
            require_operand(position.num_elements == K_RNG_POSITION_ELEMENTS, op, operand,
                            "holds {} elements, expected {}", position.num_elements,
                            K_RNG_POSITION_ELEMENTS);
            require_storage(position, op, operand);
            require_contiguous(position, op, operand);

            const SushiRuntime::Residency residency = position.storage->residency();
            require_operand(residency == SushiRuntime::Residency::Shared ||
                                residency == SushiRuntime::Residency::Host,
                            op, operand,
                            "has no host address; a position lives in Shared or Host memory");
        }
    } // namespace Dispatch

    void write_rng_position(Tensor& position, RNGPosition value)
    {
        Dispatch::require_rng_position(position, "random.write_rng_position", "position");

        std::uint64_t* words = position.data_as<std::uint64_t>();
        words[0] = value.seed;
        words[1] = value.offset;
    }
} // namespace SushiBLAS
