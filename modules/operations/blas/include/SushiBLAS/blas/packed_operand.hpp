/**************************************************************************/
/* packed_operand.hpp                                                     */
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
 * @file packed_operand.hpp
 * @brief Declares PackedOperand, the geometry of one packed triangle operand of a BLAS call.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/operand_check.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Holds the order and the layout of one packed triangle operand.
         * @see modules/operations/blas/README.md
         */
        struct PackedOperand
        {
            int64_t order = 0;                             /**< Rows, and columns, of the matrix. */
            Core::Layout layout = Core::Layout::RowMajor; /**< ROW_MAJOR packs by rows. */
        };

        /** @brief Returns how many elements the packed triangle of a matrix of order n holds. */
        [[nodiscard]] inline constexpr int64_t packed_elements(int64_t n) noexcept
        {
            return n * (n + 1) / 2;
        }

        /**
         * @brief Returns the largest order whose packed triangle fits in elements.
         * @pre elements is not negative.
         */
        [[nodiscard]] inline int64_t largest_packed_order(int64_t elements) noexcept
        {
            // The square root is an estimate; the two loops make the answer exact.
            int64_t n = static_cast<int64_t>(std::sqrt(2.0 * static_cast<double>(elements)));
            while (n > 0 && packed_elements(n) > elements)
                --n;
            while (packed_elements(n + 1) <= elements)
                ++n;
            return n;
        }

        /**
         * @brief Returns the geometry of a packed triangle operand.
         * @throws std::invalid_argument on missing storage, a rank other than 1, a strided view,
         *         or an element count that is not n * (n + 1) / 2 for any n.
         * @throws UnsupportedExecution on a dtype the tensor's device cannot compute in.
         * @see modules/operations/blas/README.md
         */
        [[nodiscard]] inline PackedOperand describe_packed_operand(const Tensor& t, const char* op,
                                                                   const char* operand)
        {
            require_storage(t, op, operand);
            require_device_dtype(t, op, operand);
            require_rank(t, 1, op, operand);
            require_contiguous(t, op, operand);

            PackedOperand p;
            p.order = largest_packed_order(t.num_elements);
            p.layout = t.layout;

            if (packed_elements(p.order) != t.num_elements)
                Detail::throw_operand_error(op, operand,
                                            "holds {} elements; a packed triangle holds "
                                            "n * (n + 1) / 2, which is {} for n = {} and {} for "
                                            "n = {}",
                                            t.num_elements, packed_elements(p.order), p.order,
                                            packed_elements(p.order + 1), p.order + 1);
            return p;
        }
    } // namespace Dispatch
} // namespace SushiBLAS
