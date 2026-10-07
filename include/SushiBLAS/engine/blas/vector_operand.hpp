/**************************************************************************/
/* vector_operand.hpp                                                     */
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
 * @file vector_operand.hpp
 * @brief Declares VectorOperand, the length and increment of one vector operand of a BLAS call.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/engine/operand_check.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Holds the element count and the distance between consecutive elements of a vector.
         * @see include/SushiBLAS/engine/blas/README.md
         */
        struct VectorOperand
        {
            int64_t length = 0;    /**< Number of elements. */
            int64_t increment = 1; /**< Distance between consecutive elements, at least 1. */
        };

        /**
         * @brief Returns the geometry of a vector operand.
         * @throws std::invalid_argument on missing storage, a non-contiguous tensor of rank
         *         above 1, or a rank-1 stride of zero or less.
         * @throws UnsupportedExecution on a dtype the tensor's device cannot compute in.
         */
        [[nodiscard]] inline VectorOperand describe_vector_operand(const Tensor& t, const char* op,
                                                                   const char* operand)
        {
            require_storage(t, op, operand);
            require_device_dtype(t, op, operand);

            VectorOperand v;
            v.length = t.num_elements;
            if (t.is_contiguous())
                return v;

            if (t.rank > 1)
                Detail::throw_operand_error(op, operand,
                                            "has rank {} and is not contiguous; a strided vector "
                                            "has rank 1",
                                            t.rank);
            if (t.strides[0] <= 0)
                Detail::throw_operand_error(op, operand,
                                            "has increment {}; a vector increment must be positive",
                                            t.strides[0]);

            v.increment = t.strides[0];
            return v;
        }

        /**
         * @brief Throws unless a vector holds the number of elements the call needs.
         * @param against Names what sets the count, such as "the columns of op(A)".
         * @throws std::invalid_argument naming both counts.
         */
        inline void require_vector_length(const VectorOperand& v, int64_t expected,
                                          const char* against, const char* op, const char* operand)
        {
            if (v.length != expected)
                Detail::throw_operand_error(op, operand, "holds {} elements, {} need {}", v.length,
                                            against, expected);
        }

        /**
         * @brief Throws unless a vector's elements are adjacent.
         * @throws std::invalid_argument naming the increment.
         */
        inline void require_unit_increment(const VectorOperand& v, const char* op,
                                           const char* operand)
        {
            if (v.increment != 1)
                Detail::throw_operand_error(op, operand,
                                            "has increment {}; this operation takes a contiguous "
                                            "vector",
                                            v.increment);
        }
    } // namespace Dispatch
} // namespace SushiBLAS
