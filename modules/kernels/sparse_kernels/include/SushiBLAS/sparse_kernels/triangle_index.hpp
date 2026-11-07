/**************************************************************************/
/* triangle_index.hpp                                                     */
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
 * @file triangle_index.hpp
 * @brief Declares the stored-triangle test and the row order of the triangular sparse kernels, shared by the host and the device.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Indexing
        {
            /**
             * @brief Returns whether entry (row, col) lies strictly inside the stored triangle:
             *        below the diagonal for a lower one, above it for an upper one.
             */
            [[nodiscard]] constexpr bool in_stored_triangle(bool lower, int64_t col,
                                                            int64_t row) noexcept
            {
                return lower ? col < row : col > row;
            }

            /**
             * @brief Returns the row a substitution visits at a step: `step` counting up, or
             *        `n - 1 - step` counting down.
             * @param ascending Whether the rows are visited from row 0.
             * @pre step lies in [0, n).
             */
            [[nodiscard]] constexpr int64_t ordered_row(int64_t step, int64_t n,
                                                        bool ascending) noexcept
            {
                return ascending ? step : n - 1 - step;
            }
        }
    }
}
