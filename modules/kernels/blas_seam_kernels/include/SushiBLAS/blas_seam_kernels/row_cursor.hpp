/**************************************************************************/
/* row_cursor.hpp                                                         */
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
 * @file row_cursor.hpp
 * @brief Declares RowCursor, the offset of an element that moves along one row of op(X).
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Holds the storage offset of one element of a row of op(X) and how it changes
         *        from one column to the next, so a walk along the row adds and never multiplies.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        struct RowCursor
        {
            int64_t offset = 0; /**< Offset of the current element from the first one stored. */
            int64_t step = 1;   /**< Elements from the current column to the next. */
            int64_t slope = 0;  /**< How much step grows from one column to the next. */

            /** @brief Moves to the next column of the row. */
            constexpr void advance() noexcept
            {
                offset += step;
                step += slope;
            }

            /** @brief Moves to the previous column of the row. */
            constexpr void retreat() noexcept
            {
                step -= slope;
                offset -= step;
            }
        };
    } // namespace Kernels
} // namespace SushiBLAS
