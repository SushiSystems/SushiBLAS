/**************************************************************************/
/* rng_position_tensor.hpp                                                */
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
 * @file rng_position_tensor.hpp
 * @brief Declares the tensor that holds a random stream position, and the call that writes it.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/rng_stream.hpp>

namespace SushiBLAS
{
    /** @brief Holds the element count of a position tensor: the seed, then the offset. */
    inline constexpr int64_t K_RNG_POSITION_ELEMENTS = 2;

    namespace Dispatch
    {
        /**
         * @brief Throws unless position can hold a stream position that a task reads when it runs.
         * @param op Name of the operation, for the message.
         * @param operand Name of the operand, for the message.
         * @throws std::invalid_argument unless position is a contiguous rank-1 INT64 tensor of
         *         K_RNG_POSITION_ELEMENTS elements in Residency::Shared or Residency::Host.
         */
        void require_rng_position(const Tensor& position, const char* op, const char* operand);
    } // namespace Dispatch

    /**
     * @brief Writes @p value into @p position on the calling thread.
     * @param position Rank-1 INT64 of K_RNG_POSITION_ELEMENTS elements, host-addressable.
     * @pre No recorded task that reads @p position is pending or running.
     * @throws std::invalid_argument on a rank, dtype, extent or residency violation.
     */
    void write_rng_position(Tensor& position, RNGPosition value);
} // namespace SushiBLAS
