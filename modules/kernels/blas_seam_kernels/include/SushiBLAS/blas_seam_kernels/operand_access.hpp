/**************************************************************************/
/* operand_access.hpp                                                     */
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
 * @file operand_access.hpp
 * @brief Declares OperandAccess, how a kernel reads element (row, col) of op(X).
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Holds the row step, the column step and the conjugation of one operand's op(X).
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        struct OperandAccess
        {
            int64_t row = 1;         /**< Elements to advance to move down one row of op(X). */
            int64_t col = 1;         /**< Elements to advance to move right one column of op(X). */
            bool conjugated = false; /**< Whether a complex element is conjugated on read. */

            /** @brief Returns the offset of element (r, c) of op(X) from the first element. */
            [[nodiscard]] constexpr int64_t index(int64_t r, int64_t c) const noexcept
            {
                return r * row + c * col;
            }

            /**
             * @brief Returns element (r, c) of op(X).
             * @pre (r, c) lies inside op(X).
             */
            template<typename T>
            [[nodiscard]] constexpr T at(const T* p, int64_t r, int64_t c) const noexcept
            {
                return conjugate_if(conjugated, p[index(r, c)]);
            }

            /** @brief Returns the access of the transpose of op(X), with the same conjugation. */
            [[nodiscard]] constexpr OperandAccess transposed() const noexcept
            {
                return OperandAccess{col, row, conjugated};
            }
        };

        /**
         * @brief Resolves a layout, a Transpose and a leading dimension into an OperandAccess.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        [[nodiscard]] inline constexpr OperandAccess operand_access(Core::Layout layout,
                                                                    Core::Transpose trans,
                                                                    int64_t ld) noexcept
        {
            const bool column_indexed =
                Core::is_transposed(trans) != (layout == Core::Layout::ColumnMajor);
            return column_indexed ? OperandAccess{1, ld, Core::is_conjugated(trans)}
                                  : OperandAccess{ld, 1, Core::is_conjugated(trans)};
        }
    } // namespace Kernels
} // namespace SushiBLAS
