/**************************************************************************/
/* blas_flags.hpp                                                         */
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
 * @file blas_flags.hpp
 * @brief Declares the mappings from a test's bool parameters to the BLAS flag enumerations.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Returns TRANS for true and NO_TRANS for false. */
        [[nodiscard]] constexpr Core::Transpose transpose_of(bool transposed) noexcept
        {
            return transposed ? Core::Transpose::Trans : Core::Transpose::NoTrans;
        }

        /** @brief Returns UPPER for true and LOWER for false. */
        [[nodiscard]] constexpr Core::Uplo uplo_of(bool upper) noexcept
        {
            return upper ? Core::Uplo::Upper : Core::Uplo::Lower;
        }

        /** @brief Returns UNIT for true and NON_UNIT for false. */
        [[nodiscard]] constexpr Core::Diag diag_of(bool unit) noexcept
        {
            return unit ? Core::Diag::Unit : Core::Diag::NonUnit;
        }

        /** @brief Returns LEFT for true and RIGHT for false. */
        [[nodiscard]] constexpr Core::Side side_of(bool left) noexcept
        {
            return left ? Core::Side::Left : Core::Side::Right;
        }

        /** @brief Returns ROW_MAJOR for true and COLUMN_MAJOR for false. */
        [[nodiscard]] constexpr Core::Layout layout_of(bool row_major) noexcept
        {
            return row_major ? Core::Layout::RowMajor : Core::Layout::ColumnMajor;
        }

        /** @brief Returns the name of the row a layout and a transpose flag make, such as "RowMajorTrans". */
        [[nodiscard]] constexpr const char* layout_transpose_label(bool row_major, bool transposed) noexcept
        {
            if (row_major)
            {
                return transposed ? "RowMajorTrans" : "RowMajorNoTrans";
            }

            return transposed ? "ColumnMajorTrans" : "ColumnMajorNoTrans";
        }

        /** @brief Returns the name of the row a layout and a triangle make, such as "ColumnMajorUpper". */
        [[nodiscard]] inline std::string layout_uplo_label(bool row_major, bool upper)
        {
            return std::string(row_major ? "RowMajor" : "ColumnMajor") + (upper ? "Upper" : "Lower");
        }

        /** @brief Returns the name of the row the four flags of a triangular routine make, such as "RowMajorLowerTransUnit". */
        [[nodiscard]] inline std::string triangular_label(bool row_major, bool upper, bool transposed, bool unit)
        {
            return layout_uplo_label(row_major, upper) + (transposed ? "Trans" : "NoTrans") + (unit ? "Unit" : "NonUnit");
        }
    } // namespace Test
} // namespace SushiBLAS
