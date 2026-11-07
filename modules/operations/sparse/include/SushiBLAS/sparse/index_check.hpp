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
 * @file index_check.hpp
 * @brief Declares the record-time checks on the index base and the arrays of a sparse operand.
 * @author Mustafa Garip
 */

#pragma once

#include <limits>
#include <cstdint>
#include <SushiBLAS/tensor/tensor.hpp>
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
         * @see modules/operations/sparse/README.md
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
         * @param array The array and the facts of its storage; a test builds the facts by hand.
         * @param part Names the array in the message, such as "column indices".
         * @throws std::invalid_argument naming the operand, the array and the rule.
         * @see modules/operations/sparse/README.md
         */
        inline void require_compressed_array(const PlacedTensor& array, int64_t expected,
                                             const char* part, const char* op, const char* operand)
        {
            const Tensor& tensor = array.tensor;
            if (tensor.rank != 1)
                Detail::throw_operand_error(op, operand, "has {} of rank {}; that array has rank 1",
                                            part, tensor.rank);
            if (tensor.num_elements != expected)
                Detail::throw_operand_error(op, operand,
                                            "has {} holding {} elements where {} are needed", part,
                                            tensor.num_elements, expected);
            if (tensor.num_elements > 1 && tensor.strides[0] != 1)
                Detail::throw_operand_error(op, operand,
                                            "has {} with increment {}; that array is contiguous",
                                            part, tensor.strides[0]);
            if (tensor.num_elements > 0 && !array.storage.present)
                Detail::throw_operand_error(op, operand,
                                            "has {} holding {} elements but no storage", part,
                                            tensor.num_elements);
        }

        /**
         * @brief Throws unless one array of a sparse operand is a contiguous rank-1 tensor of the
         *        length its operand needs, with storage when it holds elements.
         * @throws std::invalid_argument as the overload that takes the facts does.
         * @see modules/operations/sparse/README.md
         */
        inline void require_compressed_array(const Tensor& array, int64_t expected,
                                             const char* part, const char* op, const char* operand)
        {
            require_compressed_array(placed(array), expected, part, op, operand);
        }

        /**
         * @brief Throws unless an index array passes require_compressed_array, holds INT32 or
         *        INT64, and can hold the largest index its operand stores.
         * @param array The array and the facts of its storage; a test builds the facts by hand.
         * @param largest The largest value the array has to hold, index base included.
         * @throws std::invalid_argument naming the operand, the array and the rule.
         * @see modules/operations/sparse/README.md
         */
        inline void require_index_array(const PlacedTensor& array, int64_t expected,
                                        int64_t largest, const char* part, const char* op,
                                        const char* operand)
        {
            require_compressed_array(array, expected, part, op, operand);

            const Tensor& tensor = array.tensor;
            if (!Core::is_integer_dtype(tensor.dtype))
                Detail::throw_operand_error(op, operand,
                                            "has {} of dtype {}; they are INT32 or INT64", part,
                                            Core::to_string(tensor.dtype));

            const int64_t limit = (tensor.dtype == Core::DataType::Int32)
                ? static_cast<int64_t>(std::numeric_limits<int32_t>::max())
                : std::numeric_limits<int64_t>::max();
            if (largest > limit)
                Detail::throw_operand_error(op, operand,
                                            "has {} of dtype {}, which cannot hold the index {}",
                                            part, Core::to_string(tensor.dtype), largest);
        }

        /**
         * @brief Throws unless an index array passes require_compressed_array, holds INT32 or
         *        INT64, and can hold the largest index its operand stores.
         * @throws std::invalid_argument as the overload that takes the facts does.
         * @see modules/operations/sparse/README.md
         */
        inline void require_index_array(const Tensor& array, int64_t expected, int64_t largest,
                                        const char* part, const char* op, const char* operand)
        {
            require_index_array(placed(array), expected, largest, part, op, operand);
        }
    } // namespace Dispatch
} // namespace SushiBLAS
