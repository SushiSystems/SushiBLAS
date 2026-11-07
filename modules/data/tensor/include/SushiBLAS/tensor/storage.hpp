/**************************************************************************/
/* storage.hpp                                                            */
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
 * @file storage.hpp
 * @brief Declares Storage, the reference-counted allocation a tensor's elements live in.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/tensor/storage_block.hpp>
#include <SushiRuntime/core/ref_counted.hpp>

namespace SushiBLAS
{
    /**
     * @class Storage
     * @brief Holds one reference-counted allocation a tensor's elements live in, as a
     *        StorageBlock it forwards to: host memory, USM a runtime allocated or USM borrowed
     *        from another owner.
     * @see modules/README.md
     */
    class alignas(64) Storage : public SushiRuntime::RefCounted
    {
        public:
            /**
             * @brief Takes the block this storage forwards to.
             * @param block The bytes of the allocation, such as HostBlock or RuntimeBlock.
             * @throws std::runtime_error if block is null.
             */
            explicit Storage(std::unique_ptr<StorageBlock> block)
                : block_(std::move(block))
            {
                SB_THROW_IF(block_ == nullptr, "A storage needs a block");
            }

            Storage(const Storage&) = delete;
            Storage& operator=(const Storage&) = delete;

            /**
             * @brief Returns the base address of the whole allocation, ignoring every view's offset.
             * @return An address that is a multiple of K_STORAGE_ALIGNMENT, or of
             *         alignof(std::max_align_t) for a borrowed block.
             * @warning Never a tensor's data; use Tensor::data() to address elements.
             * @see modules/README.md
             */
            void* allocation_base() const noexcept { return block_->data(); }

            /**
             * @brief Returns the bytes allocated: the requested size rounded up to
             *        K_STORAGE_ALIGNMENT, or the size of the borrowed buffer.
             */
            [[nodiscard]] std::size_t size_bytes() const noexcept { return block_->size_bytes(); }

            /**
             * @brief Returns the queue of the device that owns the allocation.
             * @throws UnsupportedExecution when no device owns the allocation.
             */
            [[nodiscard]] sycl::queue& queue() const
            {
                sycl::queue* owner = block_->queue();
                if (owner == nullptr)
                    throw UnsupportedExecution("A storage with no device owner has no queue");
                return *owner;
            }

            /** @brief Returns the kind of memory the allocation lives in (Shared, Device or Host). */
            [[nodiscard]] SushiRuntime::Residency residency() const noexcept
            {
                return block_->residency();
            }

            /** @brief Returns the device queue that owns the allocation; 0 when no device does. */
            [[nodiscard]] std::size_t device_index() const noexcept
            {
                return block_->device_index();
            }

            /** @brief Returns the observer a run checks first; empty for memory the storage owns. */
            [[nodiscard]] SushiRuntime::LivenessObserver owner_liveness() const noexcept
            {
                return block_->owner_liveness();
            }

            /** @brief Returns the block this storage forwards to. */
            [[nodiscard]] const StorageBlock& block() const noexcept { return *block_; }

            /**
             * @brief Copies bytes out of the allocation into host memory and waits.
             * @param offset First byte of the allocation to copy.
             * @throws std::runtime_error if the range lies outside the allocation.
             */
            void copy_to_host(void* destination, std::size_t offset, std::size_t bytes) const
            {
                block_->copy_to_host(destination, offset, bytes);
            }

            /**
             * @brief Copies bytes from host memory into the allocation and waits.
             * @param offset First byte of the allocation to write.
             * @throws std::runtime_error if the range lies outside the allocation.
             */
            void copy_from_host(const void* source, std::size_t offset, std::size_t bytes) const
            {
                block_->copy_from_host(source, offset, bytes);
            }

        protected:
            /** @brief Frees the allocation with the block that owns it. */
            ~Storage() override = default;

        private:
            /// @brief Holds the bytes of the allocation, which free the memory with themselves.
            std::unique_ptr<StorageBlock> block_;
    };
} // namespace SushiBLAS
