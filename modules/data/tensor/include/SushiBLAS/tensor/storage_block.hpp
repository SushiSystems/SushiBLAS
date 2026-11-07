/**************************************************************************/
/* storage_block.hpp                                                      */
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
 * @file storage_block.hpp
 * @brief Declares StorageBlock, the port a Storage reads its bytes through.
 * @author Mustafa Garip
 * @see modules/data/tensor/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/region/liveness.hpp>

namespace SushiBLAS
{
    /**
     * @brief Holds the byte alignment of every owned Storage allocation base, which its size is
     *        a multiple of.
     * @see modules/README.md
     */
    inline constexpr std::size_t K_STORAGE_ALIGNMENT = 128;

    /**
     * @brief Returns n_bytes rounded up to K_STORAGE_ALIGNMENT.
     * @throws std::runtime_error if n_bytes is zero or the rounded size overflows.
     */
    [[nodiscard]] inline std::size_t rounded_storage_size(std::size_t n_bytes)
    {
        SB_THROW_IF(n_bytes == 0, "Requested size cannot be zero");
        SB_THROW_IF(n_bytes > SIZE_MAX - (K_STORAGE_ALIGNMENT - 1),
                    "Requested size {} overflows when rounded up to {} bytes",
                    n_bytes, K_STORAGE_ALIGNMENT);

        return (n_bytes + K_STORAGE_ALIGNMENT - 1) / K_STORAGE_ALIGNMENT * K_STORAGE_ALIGNMENT;
    }

    /**
     * @brief Returns whether an observer names an owner: it was made from one, even one that
     *        has since let go.
     * @param observer The observer StorageBlock::owner_liveness returned.
     * @return False for the empty observer a block that owns its memory returns.
     */
    [[nodiscard]] inline bool names_an_owner(const SushiRuntime::LivenessObserver& observer) noexcept
    {
        const SushiRuntime::LivenessObserver none;
        return observer.owner_before(none) || none.owner_before(observer);
    }

    /**
     * @class StorageBlock
     * @brief Offers the bytes of one allocation: where they are, how many, where they live and
     *        how to copy them to and from host memory.
     *
     * A Storage holds one block and forwards to it, so it names neither the host nor a runtime.
     * @see modules/data/tensor/README.md
     */
    class StorageBlock
    {
        public:
            virtual ~StorageBlock() = default;

            /** @brief Returns the first byte, or nullptr for a block that owns no memory. */
            [[nodiscard]] virtual void* data() const noexcept = 0;

            /** @brief Returns the bytes the block holds; 0 for a block that owns no memory. */
            [[nodiscard]] virtual std::size_t size_bytes() const noexcept = 0;

            /** @brief Returns the kind of memory the block lives in (Shared, Device or Host). */
            [[nodiscard]] virtual SushiRuntime::Residency residency() const noexcept = 0;

            /** @brief Returns the device queue that owns the block; 0 when no device does. */
            [[nodiscard]] virtual std::size_t device_index() const noexcept = 0;

            /**
             * @brief Returns the queue of the device that owns the block.
             * @return nullptr when no device owns the memory.
             */
            [[nodiscard]] virtual sycl::queue* queue() const = 0;

            /**
             * @brief Copies bytes from the block into host memory and returns with the copy done.
             * @param destination Host memory of at least bytes bytes.
             * @param offset First byte of the block to copy.
             * @param bytes Number of bytes to copy.
             * @throws std::runtime_error if the range lies outside the block.
             */
            virtual void copy_to_host(void* destination, std::size_t offset,
                                      std::size_t bytes) const = 0;

            /**
             * @brief Copies bytes from host memory into the block and returns with the copy done.
             * @param source Host memory of at least bytes bytes.
             * @param offset First byte of the block to write.
             * @param bytes Number of bytes to copy.
             * @throws std::runtime_error if the range lies outside the block.
             */
            virtual void copy_from_host(const void* source, std::size_t offset,
                                        std::size_t bytes) const = 0;

            /**
             * @brief Returns the observer a run checks first: it expires when the owner of
             *        borrowed memory lets go.
             * @return An empty observer for a block that owns its memory.
             */
            [[nodiscard]] virtual SushiRuntime::LivenessObserver owner_liveness() const noexcept
            {
                return {};
            }

        protected:
            /** @brief Constructs the base of an adapter. */
            StorageBlock() noexcept = default;

            /** @brief Copies the base of an adapter. */
            StorageBlock(const StorageBlock&) noexcept = default;

            /** @brief Assigns the base of an adapter. */
            StorageBlock& operator=(const StorageBlock&) noexcept = default;

            /**
             * @brief Checks that a byte range lies inside the block.
             * @throws std::runtime_error if offset + bytes passes size_bytes().
             */
            void require_range(std::size_t offset, std::size_t bytes) const
            {
                SB_THROW_IF(offset > size_bytes() || bytes > size_bytes() - offset,
                            "The range of {} bytes at offset {} lies outside a block of {} bytes",
                            bytes, offset, size_bytes());
            }
    };
} // namespace SushiBLAS
