/**************************************************************************/
/* reduce_close.hpp                                                       */
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
 * @file reduce_close.hpp
 * @brief Declares the host-callable identity write of an empty reduction.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        namespace Detail
        {
            /**
             * @brief Writes the identity as the result of a reduction over no value.
             * @param out Receives the result in out[0].
             * @see modules/foundation/graph_primitives/README.md
             */
            template <typename T>
            inline void store_identity(T* out, T identity)
            {
                out[0] = identity;
            }

            /**
             * @brief Writes the identity when the live count is zero, and nothing otherwise.
             * @param out    Holds the result the folded levels wrote.
             * @param live_n Live element count of the run.
             * @see modules/foundation/graph_primitives/README.md
             */
            template <typename T>
            inline void close_empty_reduction(T* out, std::size_t live_n, T identity)
            {
                if (live_n == 0)
                    store_identity(out, identity);
            }

        } // namespace Detail
    } // namespace GraphPrimitives
} // namespace SushiBLAS
