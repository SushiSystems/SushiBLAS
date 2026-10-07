/**************************************************************************/
/* sparse_types.hpp                                                       */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file sparse_types.hpp
 * @brief Declares IndexBase, which says what value names the first row, column or position.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiBLAS
{
    namespace Core
    {
        /**
         * @brief Selects whether the stored indices of a sparse operand count from 0 or from 1.
         * @see include/SushiBLAS/core/README.md
         */
        enum class IndexBase : uint8_t
        {
            ZERO = 221, /**< Row 0, column 0 and position 0 are stored as 0. */
            ONE = 222   /**< Row 0, column 0 and position 0 are stored as 1. */
        };

        /** @brief Returns whether value is an enumerator of IndexBase. */
        [[nodiscard]] inline constexpr bool is_valid(IndexBase value) noexcept
        {
            return value == IndexBase::ZERO || value == IndexBase::ONE;
        }

        /**
         * @brief Returns the stored value that names row 0, column 0 and position 0 under base.
         * @pre is_valid(base).
         */
        [[nodiscard]] inline constexpr int64_t first_index(IndexBase base) noexcept
        {
            return base == IndexBase::ONE ? 1 : 0;
        }

        /** @brief Returns the name of an IndexBase, or "UNKNOWN" outside the enumeration. */
        [[nodiscard]] inline constexpr const char* to_string(IndexBase value) noexcept
        {
            switch (value)
            {
                case IndexBase::ZERO:
                    return "ZERO";
                case IndexBase::ONE:
                    return "ONE";
            }
            return "UNKNOWN";
        }
    } // namespace Core
} // namespace SushiBLAS
