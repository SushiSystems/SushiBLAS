/**************************************************************************/
/* index_operands.hpp                                                     */
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
 * @file index_operands.hpp
 * @brief Declares the record-time checks the row index operations share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/operand_check.hpp>

namespace SushiBLAS
{
    namespace Gather
    {
        /**
         * @brief Throws unless index is a rank-1 INT32 or INT64 tensor with storage.
         * @throws std::invalid_argument naming the operation, the operand and the rule.
         * @see modules/operations/gather/README.md
         */
        inline void require_index_operand(const Tensor& index, const char* op, const char* operand)
        {
            Dispatch::require_rank(index, 1, op, operand);
            Dispatch::require_storage(index, op, operand);
            Dispatch::require_operand(index.dtype == Core::DataType::Int32 ||
                                          index.dtype == Core::DataType::Int64,
                                      op, operand, "has dtype {}, expected INT32 or INT64",
                                      Core::to_string(index.dtype));
        }

        /**
         * @brief Throws unless t has the given rank and storage on a device that computes its type.
         * @param rank 1 or 2.
         * @throws std::invalid_argument on a rank or storage fault; UnsupportedExecution on a dtype
         *         the device cannot compute in.
         * @see modules/operations/gather/README.md
         */
        inline void require_data_operand(const Tensor& t, int32_t rank, const char* op,
                                         const char* operand)
        {
            Dispatch::require_rank(t, rank, op, operand);
            Dispatch::require_storage(t, op, operand);
            Dispatch::require_device_dtype(t, op, operand);
        }

        /**
         * @brief Throws unless no two elements of a written tensor share an address.
         * @pre t has rank 1 or 2.
         * @throws std::invalid_argument when a stride is zero on an axis longer than one element,
         *         or the two axes of a matrix interleave.
         * @see modules/operations/gather/README.md
         */
        inline void require_distinct_elements(const Tensor& t, const char* op, const char* operand)
        {
            for (int32_t axis = 0; axis < t.rank; ++axis)
                Dispatch::require_operand(t.shape[axis] <= 1 || t.strides[axis] > 0, op, operand,
                                          "has stride {} on axis {}; a written axis needs a "
                                          "positive stride",
                                          t.strides[axis], axis);

            if (t.rank != 2 || t.shape[0] <= 1 || t.shape[1] <= 1)
                return;

            const bool rows_outside = t.strides[0] >= t.shape[1] * t.strides[1];
            const bool columns_outside = t.strides[1] >= t.shape[0] * t.strides[0];
            Dispatch::require_operand(rows_outside || columns_outside, op, operand,
                                      "has strides ({}, {}) whose elements interleave; a written "
                                      "matrix needs distinct addresses",
                                      t.strides[0], t.strides[1]);
        }

        /**
         * @brief Throws unless a written tensor and a read tensor occupy disjoint memory.
         * @throws std::invalid_argument naming both operands.
         * @see modules/operations/gather/README.md
         */
        inline void require_disjoint(const Tensor& written, const Tensor& read, const char* op,
                                     const char* written_operand, const char* read_operand)
        {
            Dispatch::require_operand(!written.region().overlaps(read.region()), op,
                                      written_operand, "overlaps '{}' in memory", read_operand);
        }

        /**
         * @brief Returns how many elements the rank-2 tensor t spans from its first element.
         * @return 0 for an empty tensor, else the offset of the last element plus one.
         * @see modules/operations/gather/README.md
         */
        [[nodiscard]] inline std::size_t spanned_elements(const Tensor& t) noexcept
        {
            if (t.num_elements == 0)
                return 0;

            return static_cast<std::size_t>((t.shape[0] - 1) * t.strides[0] +
                                            (t.shape[1] - 1) * t.strides[1] + 1);
        }
    } // namespace Gather
} // namespace SushiBLAS
