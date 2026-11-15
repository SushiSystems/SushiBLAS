/**************************************************************************/
/* tensor_copy.hpp                                                        */
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
 * @file tensor_copy.hpp
 * @brief Declares copy_tensor_to_host and copy_host_to_tensor, the byte copies of a dense tensor.
 * @author Mustafa Garip
 * @see modules/data/tensor_expectations/README.md
 */

#pragma once

#include <cstddef>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Returns how many bytes tensor's first element lies past its storage's base.
         * @throws std::runtime_error if the tensor has no storage, or is shape-only.
         */
        [[nodiscard]] inline std::size_t byte_offset_of(const Tensor& tensor)
        {
            SB_THROW_IF(tensor.shape_only, "A shape-only tensor carries no storage");
            SB_THROW_IF(tensor.storage == nullptr, "The tensor has no storage");
            return static_cast<std::size_t>(tensor.storage_offset) *
                   Core::bytes_per_element(tensor.dtype);
        }

        /**
         * @brief Copies bytes from a tensor's data into host memory and returns with the copy done.
         *
         * The storage of the tensor does the copy: host storage by memcpy, runtime storage through
         * the queue of its device, planned storage through the buffer's range copies.
         * @param bytes Bytes to copy; at most the tensor's dense byte size.
         */
        inline void copy_tensor_to_host(void* destination, const Tensor& tensor, std::size_t bytes)
        {
            tensor.storage->copy_to_host(destination, byte_offset_of(tensor), bytes);
        }

        /**
         * @brief Copies bytes from host memory into a tensor's data and returns with the copy done.
         *
         * The storage of the tensor does the copy: host storage by memcpy, runtime storage through
         * the queue of its device, planned storage through the buffer's range copies.
         * @param bytes Bytes to copy; at most the tensor's dense byte size.
         */
        inline void copy_host_to_tensor(Tensor& tensor, const void* source, std::size_t bytes)
        {
            tensor.storage->copy_from_host(source, byte_offset_of(tensor), bytes);
        }
    } // namespace Test
} // namespace SushiBLAS
