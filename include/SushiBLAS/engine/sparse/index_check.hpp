/**************************************************************************/
/* index_check.hpp                                                        */
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
 * @file index_check.hpp
 * @brief Declares the record-time checks on the index base and the arrays of a sparse operand.
 * @author Mustafa Garip
 */

#pragma once

#include <limits>
#include <cstdint>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/engine/operand_check.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Throws unless base holds an enumerator of Core::IndexBase.
         * @throws std::invalid_argument naming the operand and the numeric value.
         * @see include/SushiBLAS/engine/sparse/README.md
         */
        inline void require_index_base(Core::IndexBase base, const char* op, const char* operand)
        {
            if (!Core::is_valid(base))
                Detail::throw_operand_error(op, operand,
                                            "has the index base {}, which is not an enumerator "
                                            "of Core::IndexBase",
                                            static_cast<int>(base));
        }

        /**
         * @brief Throws unless one array of a sparse operand is a contiguous rank-1 tensor of the
         *        length its operand needs, with storage when it holds elements.
         * @param part Names the array in the message, such as "column indices".
         * @throws std::invalid_argument naming the operand, the array and the rule.
         * @see include/SushiBLAS/engine/sparse/README.md
         */
        inline void require_compressed_array(const Tensor& array, int64_t expected,
                                             const char* part, const char* op, const char* operand)
        {
            if (array.rank != 1)
                Detail::throw_operand_error(op, operand, "has {} of rank {}; that array has rank 1",
                                            part, array.rank);
            if (array.num_elements != expected)
                Detail::throw_operand_error(op, operand,
                                            "has {} holding {} elements where {} are needed", part,
                                            array.num_elements, expected);
            if (array.num_elements > 1 && array.strides[0] != 1)
                Detail::throw_operand_error(op, operand,
                                            "has {} with increment {}; that array is contiguous",
                                            part, array.strides[0]);
            if (array.num_elements > 0 && !array.storage)
                Detail::throw_operand_error(op, operand,
                                            "has {} holding {} elements but no storage", part,
                                            array.num_elements);
        }

        /**
         * @brief Throws unless an index array passes require_compressed_array, holds INT32 or
         *        INT64, and can hold the largest index its operand stores.
         * @param largest The largest value the array has to hold, index base included.
         * @throws std::invalid_argument naming the operand, the array and the rule.
         * @see include/SushiBLAS/engine/sparse/README.md
         */
        inline void require_index_array(const Tensor& array, int64_t expected, int64_t largest,
                                        const char* part, const char* op, const char* operand)
        {
            require_compressed_array(array, expected, part, op, operand);

            if (!Core::is_integer_dtype(array.dtype))
                Detail::throw_operand_error(op, operand,
                                            "has {} of dtype {}; they are INT32 or INT64", part,
                                            Core::to_string(array.dtype));

            const int64_t limit = (array.dtype == Core::DataType::Int32)
                ? static_cast<int64_t>(std::numeric_limits<int32_t>::max())
                : std::numeric_limits<int64_t>::max();
            if (largest > limit)
                Detail::throw_operand_error(op, operand,
                                            "has {} of dtype {}, which cannot hold the index {}",
                                            part, Core::to_string(array.dtype), largest);
        }
    } // namespace Dispatch
} // namespace SushiBLAS
