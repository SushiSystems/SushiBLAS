/**************************************************************************/
/* index_list.hpp                                                         */
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
 * @file index_list.hpp
 * @brief Declares IndexList, how a kernel reads the stored positions of a sparse vector.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Holds the stored positions of one sparse vector, their index base and its length.
         * @tparam Index int32_t or int64_t, the element type of the array.
         * @see kernels/README.md
         */
        template<typename Index>
        struct IndexList
        {
            int64_t count = 0;              /**< Number of stored entries. */
            int64_t extent = 0;             /**< Length of the dense vector they address. */
            int64_t first_index = 0;        /**< The stored value that names position 0. */
            const Index* indices = nullptr; /**< The position of each stored entry. */

            /**
             * @brief Returns the zero-based dense position of a stored entry.
             * @pre entry lies in [0, count).
             */
            [[nodiscard]] constexpr int64_t at(int64_t entry) const noexcept
            {
                return static_cast<int64_t>(indices[entry]) - first_index;
            }
        };
    } // namespace Kernels
} // namespace SushiBLAS
