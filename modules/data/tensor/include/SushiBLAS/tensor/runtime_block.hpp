/**************************************************************************/
/* runtime_block.hpp                                                      */
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
 * @file runtime_block.hpp
 * @brief Declares RuntimeBlock, the StorageBlock over a SushiRuntime buffer.
 * @author Mustafa Garip
 * @see modules/data/tensor/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/tensor/block_maker.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/storage_block.hpp>
#include <SushiBLAS/tensor/tensor_log_category.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    /**
     * @class RuntimeBlock
     * @brief Adapts a SushiRuntime buffer to StorageBlock: USM the runtime allocated, or USM
     *        another owner keeps alive. It copies through the queue of its device.
     * @see modules/data/tensor/README.md
     */
    class RuntimeBlock final : public StorageBlock
    {
        public:
            /** @brief Selects the constructor that borrows a buffer instead of allocating one. */
            struct Borrowed
            {

            };

            /**
             * @brief Allocates n_bytes, rounded up to K_STORAGE_ALIGNMENT, aligned to it.
             * @param n_bytes Number of bytes requested; above zero.
             * @param device_index Device queue that owns the allocation; a device of runtime.
             * @throws std::runtime_error if the size is zero or overflows, the device index is out
             *         of range, or the allocation fails.
             */
            RuntimeBlock(SushiRuntime::Runtime& runtime,
                         std::size_t n_bytes,
                         SushiRuntime::Residency residency = SushiRuntime::Residency::Shared,
                         std::size_t device_index = 0)
                : controls_(runtime.advanced())
            {
                const std::size_t size_bytes = rounded_storage_size(n_bytes);

                SB_THROW_IF(device_index >= controls_.device_count(),
                            "Device index {} is out of range; the runtime drives {} devices",
                            device_index, controls_.device_count());

                SushiRuntime::Result<SushiRuntime::Buffer<std::byte>> allocated =
                    runtime.buffer<std::byte>(size_bytes, SushiRuntime::DeviceIndex{device_index},
                                              residency, K_STORAGE_ALIGNMENT);
                SB_THROW_IF(!allocated, "Allocation failed for {} bytes: {}", size_bytes,
                            allocated.error().message);

                SB_ASSERT(K_TENSOR_LOG_CATEGORY, reinterpret_cast<std::uintptr_t>(allocated->data()) %
                                  K_STORAGE_ALIGNMENT == 0,
                          "The runtime returned an allocation that is not aligned to "
                          "K_STORAGE_ALIGNMENT");

                buffer_ = std::move(*allocated);
            }

            /**
             * @brief Views a buffer another owner keeps alive, allocating and copying nothing.
             * @param runtime The runtime the buffer belongs to.
             * @param buffer Memory that outlives every run naming it, such as the result of
             *        Interop::borrow; its size is the size of this block.
             * @throws std::runtime_error if the buffer is empty, not aligned to
             *         alignof(std::max_align_t), or on a device runtime does not drive.
             */
            RuntimeBlock(Borrowed, SushiRuntime::Runtime& runtime,
                         SushiRuntime::Buffer<std::byte>&& buffer)
                : controls_(runtime.advanced()), buffer_(std::move(buffer)), is_borrowed_(true)
            {
                SB_THROW_IF(buffer_.data() == nullptr || buffer_.size() == 0,
                            "A borrowed buffer cannot be empty");
                SB_THROW_IF(reinterpret_cast<std::uintptr_t>(buffer_.data()) %
                                    alignof(std::max_align_t) != 0,
                            "A borrowed buffer must be aligned to {} bytes",
                            alignof(std::max_align_t));
                SB_THROW_IF(buffer_.device_index().value() >= controls_.device_count(),
                            "Device index {} is out of range; the runtime drives {} devices",
                            buffer_.device_index().value(), controls_.device_count());
            }

            RuntimeBlock(const RuntimeBlock&) = delete;
            RuntimeBlock& operator=(const RuntimeBlock&) = delete;

            /** @brief Returns the first byte of the buffer. */
            [[nodiscard]] void* data() const noexcept override { return buffer_.data(); }

            /** @brief Returns the bytes allocated, or the size of the borrowed buffer. */
            [[nodiscard]] std::size_t size_bytes() const noexcept override { return buffer_.size(); }

            /** @brief Returns the residency of the buffer. */
            [[nodiscard]] SushiRuntime::Residency residency() const noexcept override
            {
                return buffer_.residency();
            }

            /** @brief Returns the device queue that owns the buffer. */
            [[nodiscard]] std::size_t device_index() const noexcept override
            {
                return buffer_.device_index().value();
            }

            /** @brief Returns the queue of the device that owns the buffer; never null. */
            [[nodiscard]] sycl::queue* queue() const override
            {
                return &controls_.queue(buffer_.device_index());
            }

            /**
             * @brief Copies bytes out of the buffer through its queue and waits.
             * @throws std::runtime_error if the range lies outside the block.
             */
            void copy_to_host(void* destination, std::size_t offset,
                              std::size_t bytes) const override
            {
                require_range(offset, bytes);
                queue()->memcpy(destination, static_cast<const std::byte*>(data()) + offset,
                                bytes).wait();
            }

            /**
             * @brief Copies bytes into the buffer through its queue and waits.
             * @throws std::runtime_error if the range lies outside the block.
             */
            void copy_from_host(const void* source, std::size_t offset,
                                std::size_t bytes) const override
            {
                require_range(offset, bytes);
                queue()->memcpy(static_cast<std::byte*>(data()) + offset, source, bytes).wait();
            }

            /** @brief Returns whether the buffer belongs to another owner. */
            [[nodiscard]] bool is_borrowed() const noexcept { return is_borrowed_; }

            /**
             * @brief Returns the observer that expires when the owner of borrowed memory lets go.
             * @return An empty observer for a block that owns its allocation.
             */
            [[nodiscard]] SushiRuntime::LivenessObserver owner_liveness() const noexcept override
            {
                if (!is_borrowed_)
                    return {};
                return SushiRuntime::On(buffer_).life;
            }

        private:
            /// @brief Holds the runtime's device view, which hands out the queue of the allocation.
            SushiRuntime::AdvancedControls controls_;

            /// @brief Holds the buffer the runtime allocated, which frees the memory with itself.
            SushiRuntime::Buffer<std::byte> buffer_;

            /// @brief Whether buffer_ views memory another owner keeps alive.
            bool is_borrowed_ = false;
    };

    /**
     * @class RuntimeBlockMaker
     * @brief Allocates RuntimeBlocks from one runtime.
     * @see modules/data/tensor/README.md
     */
    class RuntimeBlockMaker final : public BlockMaker
    {
        public:
            /**
             * @brief Binds the maker to the runtime that allocates.
             * @param runtime Performs every allocation; it must outlive the maker.
             */
            explicit RuntimeBlockMaker(SushiRuntime::Runtime& runtime) noexcept
                : runtime_(runtime)
            {

            }

            /** @brief Allocates a RuntimeBlock on the runtime; the throws are RuntimeBlock's. */
            [[nodiscard]] std::unique_ptr<StorageBlock> make(
                std::size_t n_bytes, SushiRuntime::Residency residency,
                std::size_t device_index) const override
            {
                return std::make_unique<RuntimeBlock>(runtime_, n_bytes, residency, device_index);
            }

            /** @brief Returns the number of devices the runtime drives. */
            [[nodiscard]] std::size_t device_count() const override
            {
                return runtime_.advanced().device_count();
            }

            /** @brief Returns the runtime that allocates. */
            [[nodiscard]] SushiRuntime::Runtime& runtime() const override { return runtime_; }

        private:
            /// @brief Holds the runtime that allocates.
            SushiRuntime::Runtime& runtime_;
    };

    /**
     * @brief Allocates a Storage over a runtime buffer, n_bytes rounded up to K_STORAGE_ALIGNMENT.
     * @param runtime Allocates the buffer the Storage owns.
     * @param n_bytes Number of bytes requested; above zero.
     * @param residency Where the memory lives.
     * @param device_index Device queue that owns the allocation; a device of runtime.
     * @throws std::runtime_error as the RuntimeBlock allocating constructor states.
     */
    [[nodiscard]] inline SushiRuntime::sushi_ptr<Storage> make_runtime_storage(
        SushiRuntime::Runtime& runtime, std::size_t n_bytes,
        SushiRuntime::Residency residency = SushiRuntime::Residency::Shared,
        std::size_t device_index = 0)
    {
        return SushiRuntime::make_sushi<Storage>(
            std::make_unique<RuntimeBlock>(runtime, n_bytes, residency, device_index));
    }

    /**
     * @brief Makes a Storage that views a buffer another owner keeps alive, allocating and
     *        copying nothing.
     * @param runtime The runtime the buffer belongs to.
     * @param buffer Memory that outlives every run naming it; its size is the size of the Storage.
     * @throws std::runtime_error as the RuntimeBlock borrowing constructor states.
     */
    [[nodiscard]] inline SushiRuntime::sushi_ptr<Storage> borrow_runtime_storage(
        SushiRuntime::Runtime& runtime, SushiRuntime::Buffer<std::byte>&& buffer)
    {
        return SushiRuntime::make_sushi<Storage>(std::make_unique<RuntimeBlock>(
            RuntimeBlock::Borrowed{}, runtime, std::move(buffer)));
    }
} // namespace SushiBLAS
