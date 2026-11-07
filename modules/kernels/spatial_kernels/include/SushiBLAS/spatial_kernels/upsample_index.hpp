/**************************************************************************/
/* upsample_index.hpp                                                     */
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
 * @file upsample_index.hpp
 * @brief Declares the index math of nearest-neighbour upsampling, shared by the host and the device.
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
            /** @brief Returns the input coordinate along one axis that output coordinate `out_pos` repeats: `out_pos / scale`. */
            [[nodiscard]] constexpr int64_t upsample_source(int64_t out_pos, int64_t scale) noexcept
            {
                return out_pos / scale;
            }

            /** @brief Returns the output coordinate `step` places into the block of input coordinate `in_pos`: `in_pos * scale + step`. */
            [[nodiscard]] constexpr int64_t upsample_block(int64_t in_pos, int64_t scale,
                                                           int64_t step) noexcept
            {
                return in_pos * scale + step;
            }
        }
    }
}
