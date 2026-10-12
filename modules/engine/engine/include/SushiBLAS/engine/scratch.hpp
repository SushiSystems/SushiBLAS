/**************************************************************************/
/* scratch.hpp                                                            */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file scratch.hpp
 * @brief Declares ScratchBuffer, the lease on one pooled device scratch block.
 * @author Mustafa Garip
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <utility>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/scratch_shelf.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>

namespace SushiBLAS
{
    /**
     * @class ScratchBuffer
     * @brief Leases one device scratch block; the block returns to its pool when the last copy dies.
     * @see modules/engine/engine/README.md
     */
    class ScratchBuffer final
    {
        public:
            /** @brief Builds an empty handle that leases nothing. */
            ScratchBuffer() noexcept = default;

            /**
             * @brief Leases at least bytes of device memory from pool, or nothing when bytes is zero.
             * @param device_index Device queue that owns the block: the operand's
             *        Storage::device_index().
             * @throws std::runtime_error if the allocation of a new block fails.
             */
            ScratchBuffer(ScratchPool& pool, std::size_t bytes, std::size_t device_index)
                : bytes_(bytes)
            {
                if (bytes_ == 0)
                    return;

                block_ = pool.shelf()->take(bytes_, device_index);
                block_->shelf = pool.shelf();
                block_->handles.store(1, std::memory_order_relaxed);
            }

            /** @brief Puts one more handle on other's block. */
            ScratchBuffer(const ScratchBuffer& other) noexcept
                : block_(other.block_), bytes_(other.bytes_)
            {
                if (block_ != nullptr)
                    block_->handles.fetch_add(1, std::memory_order_relaxed);
            }

            /** @brief Takes other's handle and leaves other empty. */
            ScratchBuffer(ScratchBuffer&& other) noexcept
                : block_(std::exchange(other.block_, nullptr)),
                  bytes_(std::exchange(other.bytes_, std::size_t{0}))
            {

            }

            /** @brief Releases this handle and puts one more on other's block. */
            ScratchBuffer& operator=(const ScratchBuffer& other) noexcept
            {
                ScratchBuffer copy(other);
                swap(copy);
                return *this;
            }

            /** @brief Releases this handle and takes other's. */
            ScratchBuffer& operator=(ScratchBuffer&& other) noexcept
            {
                ScratchBuffer moved(std::move(other));
                swap(moved);
                return *this;
            }

            /** @brief Releases this handle; the last one returns the block to its shelf. */
            ~ScratchBuffer()
            {
                if (block_ == nullptr)
                    return;

                if (block_->handles.fetch_sub(1, std::memory_order_acq_rel) != 1)
                    return;

                // The shelf pointer leaves the block first, so an idle block never owns its shelf.
                const SushiRuntime::sushi_ptr<Detail::ScratchShelf> shelf = std::move(block_->shelf);
                shelf->give_back(block_);
            }

            /** @brief Returns the base address, or null when the handle leases nothing. */
            [[nodiscard]] void* data() const noexcept
            {
                return block_ != nullptr ? block_->storage->allocation_base() : nullptr;
            }

            /** @brief Returns the usable size in bytes: what the caller asked for, 0 when empty. */
            [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

            /** @brief Returns true when the handle leases no block. */
            [[nodiscard]] bool is_empty() const noexcept { return bytes_ == 0; }

            /**
             * @brief Returns the region a task that writes the scratch declares.
             * @pre is_empty() is false.
             */
            [[nodiscard]] SushiRuntime::BufferRegion region() const noexcept
            {
                return {data(), 0, bytes_};
            }

            /**
             * @brief Returns the allocation behind the leased block, or null when the handle
             *        leases nothing.
             */
            [[nodiscard]] SushiRuntime::sushi_ptr<Storage> storage() const noexcept
            {
                return block_ != nullptr ? block_->storage : SushiRuntime::sushi_ptr<Storage>();
            }

        private:
            /** @brief Exchanges the handles of two buffers. */
            void swap(ScratchBuffer& other) noexcept
            {
                std::swap(block_, other.block_);
                std::swap(bytes_, other.bytes_);
            }

            /// @brief The leased block, or null when bytes_ is zero.
            Detail::ScratchBlock* block_ = nullptr;

            /// @brief The size the caller asked for, in bytes.
            std::size_t bytes_ = 0;
    };
} // namespace SushiBLAS
