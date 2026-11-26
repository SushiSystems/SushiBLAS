/**************************************************************************/
/* io_tensor_access.hpp                                                   */
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
 * @file io_tensor_access.hpp
 * @brief Declares how the IO units read the payload size, residency and queue of a tensor.
 * @author Mustafa Garip
 * @see modules/operations/io/README.md
 */

#pragma once

#include <cstddef>
#include <SushiBLAS/tensor/tensor.hpp>

namespace SushiBLAS
{
    namespace Detail
    {
        /** @brief Returns the number of bytes the tensor's elements occupy. */
        inline size_t payload_bytes(const Tensor& t)
        {
            return static_cast<size_t>(t.num_elements) * Core::bytes_per_element(t.dtype);
        }

        /** @brief Reports whether the host reaches the tensor's storage only through the queue. */
        inline bool is_device_allocated(const Tensor& t)
        {
            return t.storage->residency() == SushiRuntime::Residency::Device;
        }

        /**
         * @brief Copies bytes of the tensor's payload to host memory through its storage and
         *        waits, so the runtime sees the copy.
         * @param payload_offset First payload byte to copy.
         * @pre t has storage, t is dense, and payload_offset + bytes is at most payload_bytes(t).
         * @throws std::runtime_error if the storage fails the copy.
         */
        inline void read_payload_bytes(const Tensor& t, size_t payload_offset, void* destination,
                                       size_t bytes)
        {
            const size_t base = static_cast<size_t>(t.storage_offset) *
                                Core::bytes_per_element(t.dtype);
            t.storage->copy_to_host(destination, base + payload_offset, bytes);
        }

        /**
         * @brief Copies host bytes into the tensor's payload through its storage and waits, so
         *        the runtime sees the copy.
         * @param payload_offset First payload byte to write.
         * @pre t has storage, t is dense, and payload_offset + bytes is at most payload_bytes(t).
         * @throws std::runtime_error if the storage fails the copy.
         */
        inline void write_payload_bytes(const Tensor& t, size_t payload_offset, const void* source,
                                        size_t bytes)
        {
            const size_t base = static_cast<size_t>(t.storage_offset) *
                                Core::bytes_per_element(t.dtype);
            t.storage->copy_from_host(source, base + payload_offset, bytes);
        }
    } // namespace Detail
} // namespace SushiBLAS
