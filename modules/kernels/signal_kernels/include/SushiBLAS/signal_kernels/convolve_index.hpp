/**************************************************************************/
/* convolve_index.hpp                                                     */
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
 * @file convolve_index.hpp
 * @brief Declares the sample and tap indexing of the valid-mode convolution, shared by the host and the device.
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
            /** @brief Returns the samples one row of a valid-mode convolution writes: `L - K + 1`. */
            [[nodiscard]] constexpr int64_t convolve_out_length(int64_t signal_length,
                                                                int64_t tap_count) noexcept
            {
                return signal_length - tap_count + 1;
            }

            /** @brief Holds one output sample's position: the row and the sample in the row. */
            struct ConvolveCoord
            {
                int64_t row = 0;    /**< Dense row. */
                int64_t n = 0;      /**< Sample in the row. */
            };

            /**
             * @brief Splits a flat output index, sample fastest, into `(row, n)`.
             * @param out_length Must be positive.
             */
            [[nodiscard]] constexpr ConvolveCoord convolve_coord(int64_t flat,
                                                                 int64_t out_length) noexcept
            {
                return ConvolveCoord{flat / out_length, flat % out_length};
            }

            /** @brief Returns the signal sample tap `k` multiplies for output sample `n`: `n + K - 1 - k`. */
            [[nodiscard]] constexpr int64_t convolve_source(int64_t n, int64_t tap_count,
                                                            int64_t k) noexcept
            {
                return n + tap_count - 1 - k;
            }
        }
    }
}
