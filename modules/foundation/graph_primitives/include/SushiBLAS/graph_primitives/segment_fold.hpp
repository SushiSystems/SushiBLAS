/**************************************************************************/
/* segment_fold.hpp                                                       */
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
 * @file segment_fold.hpp
 * @brief Declares the host-callable fold of one segment of a ragged array.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <SushiBLAS/graph_primitives/fixed_order_fold.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        namespace Detail
        {
            /**
             * @brief Folds segment s of values left to right; an empty segment yields identity.
             * @param offsets Segment s covers values[offsets[s], offsets[s + 1]).
             * @see modules/foundation/graph_primitives/README.md
             */
            template <typename T, class Combine>
            [[nodiscard]] inline T fold_segment(const Combine& op, const T* values,
                                                const std::uint32_t* offsets, std::size_t s,
                                                T identity)
            {
                const std::size_t begin = offsets[s];
                const std::size_t end = offsets[s + 1];
                return (end > begin) ? fold_tile(op, values, begin, end) : identity;
            }

        } // namespace Detail
    } // namespace GraphPrimitives
} // namespace SushiBLAS
