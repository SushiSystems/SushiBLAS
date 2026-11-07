/**************************************************************************/
/* rhs_index.hpp                                                          */
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
 * @file rhs_index.hpp
 * @brief Declares the split of a flat product index into a matrix row and a right-hand side, shared by the host and the device.
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
            /** @brief Holds one element of a row-major product: the matrix row and the right-hand side. */
            struct RhsCoord
            {
                int64_t row = 0; /**< Row of the sparse matrix. */
                int64_t rhs = 0; /**< Column of the dense operand. */
            };

            /**
             * @brief Splits a flat index, right-hand side fastest, into `(row, rhs)`.
             * @param nrhs Must be positive.
             */
            [[nodiscard]] constexpr RhsCoord rhs_coord(int64_t flat, int64_t nrhs) noexcept
            {
                return RhsCoord{flat / nrhs, flat % nrhs};
            }
        }
    }
}
