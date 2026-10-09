/**************************************************************************/
/* structure.hpp                                                          */
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
 * @file structure.hpp
 * @brief Defines the host checks of the structure a sparse kernel takes on trust.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/index_list.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Names the first rule a sparse structure breaks, or none.
             * @see kernels/README.md
             */
            enum class StructureDefect : uint8_t
            {
                None,               /**< The structure keeps every rule. */
                FirstOffset,       /**< The first row offset is not the index base. */
                DecreasingOffset,  /**< A row ends before it starts. */
                LastOffset,        /**< The last row offset does not name the entry count. */
                IndexOutOfRange, /**< A column or a position lies outside the operand. */
                RepeatedIndex      /**< A sparse vector stores one position twice. */
            };

            /**
             * @brief Returns the first rule the arrays of a CSR matrix break.
             * @pre row_offsets holds rows + 1 elements and column_indices holds nnz.
             * @see kernels/README.md
             */
            template<typename Index>
            [[nodiscard]] StructureDefect find_defect(const CsrPattern<Index>& pattern) noexcept
            {
                if (pattern.row_begin(0) != 0)
                    return StructureDefect::FirstOffset;

                for (int64_t row = 0; row < pattern.rows; ++row)
                {
                    if (pattern.row_end(row) < pattern.row_begin(row))
                        return StructureDefect::DecreasingOffset;
                }

                if (pattern.row_begin(pattern.rows) != pattern.nnz)
                    return StructureDefect::LastOffset;

                for (int64_t entry = 0; entry < pattern.nnz; ++entry)
                {
                    const int64_t column = pattern.column(entry);
                    if (column < 0 || column >= pattern.cols)
                        return StructureDefect::IndexOutOfRange;
                }

                return StructureDefect::None;
            }

            /**
             * @brief Returns the first rule the stored positions of a sparse vector break.
             * @pre indices holds count elements.
             * @see kernels/README.md
             */
            template<typename Index>
            [[nodiscard]] StructureDefect find_defect(const IndexList<Index>& positions)
            {
                std::vector<bool> seen(static_cast<std::size_t>(positions.extent), false);
                for (int64_t entry = 0; entry < positions.count; ++entry)
                {
                    const int64_t position = positions.at(entry);
                    if (position < 0 || position >= positions.extent)
                        return StructureDefect::IndexOutOfRange;
                    if (seen[static_cast<std::size_t>(position)])
                        return StructureDefect::RepeatedIndex;
                    seen[static_cast<std::size_t>(position)] = true;
                }

                return StructureDefect::None;
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
