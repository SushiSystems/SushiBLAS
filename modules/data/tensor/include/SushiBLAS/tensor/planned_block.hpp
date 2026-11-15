/**************************************************************************/
/* planned_block.hpp                                                      */
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
 * @file planned_block.hpp
 * @brief Declares PlannedBlock, the StorageBlock over a SushiRuntime buffer the runtime places.
 * @author Mustafa Garip
 * @see modules/data/tensor/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/tensor/storage_block.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    /**
     * @class PlannedBlock
     * @brief Adapts a Residency::Planned SushiRuntime buffer to StorageBlock: no host address,
     *        no queue, host copies through the buffer's read_range() and write_range().
     * @see modules/data/tensor/README.md
     */
    class PlannedBlock final : public StorageBlock
    {
        public:
            /**
             * @brief Makes a planned buffer of n_bytes rounded up to K_STORAGE_ALIGNMENT.
             * @param runtime The runtime that places the buffer; it must outlive the block.
             * @param n_bytes Number of bytes requested; above zero.
             * @throws std::runtime_error if the size is zero or overflows, or the runtime refuses.
             */
            PlannedBlock(SushiRuntime::Runtime& runtime, std::size_t n_bytes)
                : buffer_(take(runtime.buffer<std::byte>(rounded_storage_size(n_bytes),
                                                         SushiRuntime::Residency::Planned),
                               "planned buffer"))
            {

            }

            /**
             * @brief Makes a planned buffer whose one valid copy is n_bytes of a file from offset.
             * @param path The file; it must not change while the buffer lives.
             * @param offset First byte of the file the block holds.
             * @param n_bytes Number of bytes the block holds; above zero and not rounded.
             * @throws std::runtime_error if the size is zero, or the file cannot be read or is
             *         shorter than the range.
             */
            PlannedBlock(SushiRuntime::Runtime& runtime, const std::string& path,
                         std::uint64_t offset, std::size_t n_bytes)
                : buffer_(take(runtime.buffer_from_file<std::byte>(path, offset,
                                                                   nonzero(n_bytes)),
                               "file buffer"))
            {

            }

            PlannedBlock(const PlannedBlock&) = delete;
            PlannedBlock& operator=(const PlannedBlock&) = delete;

            /** @brief Returns nullptr: the runtime may move the bytes, so they have no address. */
            [[nodiscard]] void* data() const noexcept override { return nullptr; }

            /** @brief Returns the bytes the buffer holds. */
            [[nodiscard]] std::size_t size_bytes() const noexcept override { return buffer_.size(); }

            /** @brief Returns Residency::Planned. */
            [[nodiscard]] SushiRuntime::Residency residency() const noexcept override
            {
                return buffer_.residency();
            }

            /** @brief Returns no device; the runtime chooses where the bytes live. */
            [[nodiscard]] std::optional<std::size_t> device_index() const noexcept override
            {
                return std::nullopt;
            }

            /** @brief Returns nullptr; no device owns a planned buffer. */
            [[nodiscard]] sycl::queue* queue() const override { return nullptr; }

            /**
             * @brief Copies bytes out of the buffer through read_range() and returns with the copy done.
             * @throws std::runtime_error if the range lies outside the block or the read fails.
             */
            void copy_to_host(void* destination, std::size_t offset,
                              std::size_t bytes) const override
            {
                require_range(offset, bytes);
                if (bytes == 0)
                    return;
                const SushiRuntime::Result<std::vector<std::byte>> read =
                    buffer_.read_range(SushiRuntime::ElementRange{offset, bytes});
                SB_THROW_IF(!read, "Reading {} bytes at offset {} of a planned buffer failed: {}",
                            bytes, offset, read.error().message);
                std::memcpy(destination, read->data(), bytes);
            }

            /**
             * @brief Copies bytes into the buffer through write_range() and returns with the copy done.
             * @throws std::runtime_error if the range lies outside the block or the write fails.
             */
            void copy_from_host(const void* source, std::size_t offset,
                                std::size_t bytes) const override
            {
                require_range(offset, bytes);
                if (bytes == 0)
                    return;
                const SushiRuntime::Result<void> written = buffer_.write_range(
                    SushiRuntime::ElementRange{offset, bytes},
                    SushiRuntime::span<const std::byte>(static_cast<const std::byte*>(source),
                                                        bytes));
                SB_THROW_IF(!written, "Writing {} bytes at offset {} of a planned buffer failed: {}",
                            bytes, offset, written.error().message);
            }

            /** @brief Returns the handle the runtime minted for the buffer. */
            [[nodiscard]] SushiRuntime::GlobalHandle global_handle() const noexcept override
            {
                return buffer_.handle();
            }

        private:
            /**
             * @brief Returns the buffer a runtime call made.
             * @throws std::runtime_error with the runtime's message when the call failed.
             */
            static SushiRuntime::Buffer<std::byte> take(
                SushiRuntime::Result<SushiRuntime::Buffer<std::byte>>&& made, const char* what)
            {
                SB_THROW_IF(!made, "The runtime refused a {}: {}", what, made.error().message);
                return std::move(*made);
            }

            /**
             * @brief Returns n_bytes.
             * @throws std::runtime_error if n_bytes is zero.
             */
            static std::size_t nonzero(std::size_t n_bytes)
            {
                SB_THROW_IF(n_bytes == 0, "Requested size cannot be zero");
                return n_bytes;
            }

            /// @brief Holds the planned buffer, which releases its handle with itself.
            SushiRuntime::Buffer<std::byte> buffer_;
    };
} // namespace SushiBLAS
