/**************************************************************************/
/* composite_cost.hpp                                                     */
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
 * @file composite_cost.hpp
 * @brief Declares the cost of operations a caller composes from several recorded tasks.
 * @author Mustafa Garip
 * @see modules/foundation/cost/README.md
 */

#pragma once

#include <cstdint>

namespace SushiBLAS
{
    /**
     * @brief Returns the floating-point operations of a softmax over rows of equal length.
     *
     * Per row: the maximum (cols - 1 compares), the shift (cols), the exponential (cols), the
     * sum (cols - 1) and the division (cols), which is `5 * cols - 2`. A row or column count
     * below 1 costs 0.
     * @see modules/foundation/cost/README.md
     */
    [[nodiscard]] constexpr std::uint64_t softmax_flops(std::int64_t rows,
                                                        std::int64_t cols) noexcept
    {
        if (rows < 1 || cols < 1)
            return 0;
        return static_cast<std::uint64_t>(rows) * (5U * static_cast<std::uint64_t>(cols) - 2U);
    }
} // namespace SushiBLAS
