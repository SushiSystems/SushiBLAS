/**************************************************************************/
/* host_block.hpp                                                         */
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
 * @file host_block.hpp
 * @brief Declares HostBlock, aligned host memory that no SushiRuntime::Runtime owns, and the
 *        adapters that make it a Storage and a TensorFactory source.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/tensor/block_maker.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/storage_block.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>

namespace SushiBLAS
{
    /**
     * @class HostBlock
     * @brief Adapts ordinary host memory to StorageBlock: owns one aligned block, frees it with
     *        itself and copies by std::memcpy.
     * @see modules/data/tensor/README.md
     */
    class HostBlock final : public StorageBlock
    {
        public:
            /** @brief Constructs a block that owns no memory. */
            HostBlock() noexcept = default;

            /**
             * @brief Allocates n_bytes of host memory at an address aligned to alignment.
             * @param n_bytes Number of bytes; above zero.
             * @param alignment A power of two.
             * @throws std::bad_alloc if the allocation fails.
             */
            HostBlock(std::size_t n_bytes, std::size_t alignment)
                : data_(::operator new(n_bytes, std::align_val_t{alignment})),
                  size_bytes_(n_bytes),
                  alignment_(alignment)
            {

            }

            HostBlock(const HostBlock&) = delete;
            HostBlock& operator=(const HostBlock&) = delete;

            /** @brief Takes the memory of other, which owns none afterwards. */
            HostBlock(HostBlock&& other) noexcept
                : data_(other.data_), size_bytes_(other.size_bytes_), alignment_(other.alignment_)
            {
                other.data_ = nullptr;
                other.size_bytes_ = 0;
            }

            /** @brief Frees the memory it owns, then takes the memory of other. */
            HostBlock& operator=(HostBlock&& other) noexcept
            {
                if (this != &other)
                {
                    release();
                    data_ = other.data_;
                    size_bytes_ = other.size_bytes_;
                    alignment_ = other.alignment_;
                    other.data_ = nullptr;
                    other.size_bytes_ = 0;
                }
                return *this;
            }

            /** @brief Frees the memory it owns. */
            ~HostBlock() override
            {
                release();
            }

            /** @brief Returns the first byte, or nullptr for a block that owns no memory. */
            [[nodiscard]] void* data() const noexcept override { return data_; }

            /** @brief Returns the bytes owned; 0 for a block that owns no memory. */
            [[nodiscard]] std::size_t size_bytes() const noexcept override { return size_bytes_; }

            /** @brief Returns Residency::Host. */
            [[nodiscard]] SushiRuntime::Residency residency() const noexcept override
            {
                return SushiRuntime::Residency::Host;
            }

            /** @brief Returns 0; no device owns host memory. */
            [[nodiscard]] std::size_t device_index() const noexcept override { return 0; }

            /** @brief Returns nullptr; no device owns host memory. */
            [[nodiscard]] sycl::queue* queue() const override { return nullptr; }

            /**
             * @brief Copies bytes out of the block by std::memcpy.
             * @throws std::runtime_error if the range lies outside the block.
             */
            void copy_to_host(void* destination, std::size_t offset,
                              std::size_t bytes) const override
            {
                require_range(offset, bytes);
                if (bytes != 0)
                    std::memcpy(destination, static_cast<const std::byte*>(data_) + offset, bytes);
            }

            /**
             * @brief Copies bytes into the block by std::memcpy.
             * @throws std::runtime_error if the range lies outside the block.
             */
            void copy_from_host(const void* source, std::size_t offset,
                                std::size_t bytes) const override
            {
                require_range(offset, bytes);
                if (bytes != 0)
                    std::memcpy(static_cast<std::byte*>(data_) + offset, source, bytes);
            }

        private:
            /** @brief Frees the memory and leaves the block empty. */
            void release() noexcept
            {
                if (data_ != nullptr)
                    ::operator delete(data_, std::align_val_t{alignment_});
                data_ = nullptr;
            }

            /// @brief The first byte of the block; null when it owns none.
            void* data_ = nullptr;

            /// @brief The bytes owned.
            std::size_t size_bytes_ = 0;

            /// @brief The alignment the block was allocated with, which its release needs.
            std::size_t alignment_ = alignof(std::max_align_t);
    };

    /**
     * @class HostBlockMaker
     * @brief Allocates HostBlocks, rounded and aligned to K_STORAGE_ALIGNMENT, and has no runtime.
     * @see modules/data/tensor/README.md
     */
    class HostBlockMaker final : public BlockMaker
    {
        public:
            /**
             * @brief Allocates host memory no runtime owns; residency and device_index are ignored.
             * @param n_bytes Number of bytes requested; above zero.
             * @throws std::runtime_error if the size is zero or overflows.
             * @throws std::bad_alloc if the allocation fails.
             */
            [[nodiscard]] std::unique_ptr<StorageBlock> make(
                std::size_t n_bytes, SushiRuntime::Residency, std::size_t) const override
            {
                return std::make_unique<HostBlock>(rounded_storage_size(n_bytes),
                                                   K_STORAGE_ALIGNMENT);
            }

            /** @brief Returns 1; host memory has one place. */
            [[nodiscard]] std::size_t device_count() const override { return 1; }

            /** @brief Always throws UnsupportedExecution; host memory has no runtime. */
            [[noreturn]] SushiRuntime::Runtime& runtime() const override
            {
                throw UnsupportedExecution("A host TensorFactory has no runtime");
            }
    };

    /**
     * @brief Allocates a Storage over host memory no runtime owns, n_bytes rounded up to
     *        K_STORAGE_ALIGNMENT, at an address aligned to it.
     * @param n_bytes Number of bytes requested; above zero.
     * @throws std::runtime_error if the size is zero or overflows.
     * @throws std::bad_alloc if the allocation fails.
     */
    [[nodiscard]] inline SushiRuntime::sushi_ptr<Storage> make_host_storage(std::size_t n_bytes)
    {
        return SushiRuntime::make_sushi<Storage>(
            HostBlockMaker().make(n_bytes, SushiRuntime::Residency::Host, 0));
    }
} // namespace SushiBLAS
