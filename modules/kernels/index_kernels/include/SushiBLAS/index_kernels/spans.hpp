/**************************************************************************/
/* spans.hpp                                                              */
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
 * @file spans.hpp
 * @brief Declares MatrixSpan and is_in_range, the geometry every index kernel addresses by.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Holds the extents and the element strides of one rank-2 operand.
         * @see modules/kernels/index_kernels/README.md
         */
        struct MatrixSpan
        {
            int64_t rows = 0;       /**< Extent of the first axis. */
            int64_t cols = 0;       /**< Extent of the second axis. */
            int64_t row_stride = 0; /**< Elements between two consecutive rows. */
            int64_t col_stride = 0; /**< Elements between two consecutive columns. */

            /** @brief Returns the offset of element (r, c) from the first element. */
            [[nodiscard]] constexpr int64_t offset(int64_t r, int64_t c) const noexcept
            {
                return r * row_stride + c * col_stride;
            }
        };

        /**
         * @brief Returns whether an index of either width selects a position of `[0, extent)`.
         * @see modules/kernels/index_kernels/README.md
         */
        template<typename Index>
        [[nodiscard]] constexpr bool is_in_range(Index index, int64_t extent) noexcept
        {
            const int64_t position = static_cast<int64_t>(index);
            return position >= 0 && position < extent;
        }
    } // namespace Kernels
} // namespace SushiBLAS
