/**************************************************************************/
/* broadcast_operand.hpp                                                  */
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
 * @file broadcast_operand.hpp
 * @brief Declares BroadcastAxis and BroadcastOperand, which name one vector of a broadcast call.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/tensor/tensor.hpp>

namespace SushiBLAS
{
    /** @brief Names which index of the matrix selects a vector's element. */
    enum class BroadcastAxis : std::uint8_t
    {
        Leading, /**< One value per outer index: `v[o]` meets every element of row o. */
        Trailing /**< One value per inner index: `v[n]` meets element n of every row. */
    };

    /** @brief Holds one vector of a broadcast call and the index that selects from it. */
    struct BroadcastOperand
    {
        /** @brief Holds the vector; it is never null in a recorded call. */
        const Tensor* vector = nullptr;

        /** @brief Holds the index that selects from the vector. */
        BroadcastAxis axis = BroadcastAxis::Trailing;
    };
} // namespace SushiBLAS
