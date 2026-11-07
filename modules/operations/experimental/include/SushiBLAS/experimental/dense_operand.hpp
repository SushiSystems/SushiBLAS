/**************************************************************************/
/* dense_operand.hpp                                                      */
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
 * @file dense_operand.hpp
 * @brief Declares the record-time operand checks the transforms and dense solvers share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/operand_check.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        /**
         * @brief Throws unless t has storage, is dense, holds @p elements and lives on a device
         *        that computes its dtype.
         * @param placed The tensor with the facts of its storage.
         * @throws std::invalid_argument naming op and operand; UnsupportedExecution for fp64.
         */
        inline void require_dense_elements(const Dispatch::PlacedTensor& placed,
                                           std::size_t elements, const char* op,
                                           const char* operand)
        {
            Dispatch::require_storage(placed.tensor, placed.storage, op, operand);
            Dispatch::require_contiguous(placed.tensor, op, operand);
            Dispatch::require_extent(placed.tensor.num_elements, static_cast<int64_t>(elements),
                                     "elements", op, operand);
            Dispatch::require_device_dtype(placed.tensor, placed.storage, op, operand);
        }

        /**
         * @brief Throws unless t has storage, is dense, holds @p elements and lives on a device
         *        that computes its dtype.
         * @throws std::invalid_argument naming op and operand; UnsupportedExecution for fp64.
         */
        inline void require_dense_elements(const Tensor& t, std::size_t elements, const char* op,
                                           const char* operand)
        {
            require_dense_elements(Dispatch::placed(t), elements, op, operand);
        }

        /** @brief Throws unless t is FLOAT32 or FLOAT64. */
        inline void require_float_dtype(const Tensor& t, const char* op, const char* operand)
        {
            Dispatch::require_operand(t.dtype == Core::DataType::Float32 ||
                                          t.dtype == Core::DataType::Float64,
                                      op, operand, "has dtype {}; FLOAT32 or FLOAT64 is required",
                                      Core::to_string(t.dtype));
        }

        /** @brief Throws unless t is COMPLEX32 or COMPLEX64. */
        inline void require_complex_dtype(const Tensor& t, const char* op, const char* operand)
        {
            Dispatch::require_operand(t.dtype == Core::DataType::Complex32 ||
                                          t.dtype == Core::DataType::Complex64,
                                      op, operand,
                                      "has dtype {}; COMPLEX32 or COMPLEX64 is required",
                                      Core::to_string(t.dtype));
        }

        /** @brief Returns the real dtype of a complex one's parts, and a real dtype unchanged. */
        [[nodiscard]] inline Core::DataType scalar_dtype_of(Core::DataType dtype) noexcept
        {
            if (dtype == Core::DataType::Complex32)
                return Core::DataType::Float32;
            if (dtype == Core::DataType::Complex64)
                return Core::DataType::Float64;
            return dtype;
        }
    } // namespace Internal
} // namespace SushiBLAS
