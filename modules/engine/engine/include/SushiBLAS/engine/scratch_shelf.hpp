/**************************************************************************/
/* scratch_shelf.hpp                                                      */
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
 * @file scratch_shelf.hpp
 * @brief Declares the shelf of idle device scratch blocks and the block it lends.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <mutex>
#include <atomic>
#include <memory>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    namespace Detail
    {
        /** @brief Holds the smallest pooled block size in bytes. */
        inline constexpr std::size_t K_SCRATCH_MIN_BLOCK_BYTES = 128;

        /** @brief Holds the largest pooled block size in bytes; a larger request is not pooled. */
        inline constexpr std::size_t K_SCRATCH_MAX_BLOCK_BYTES = 4096;

        /** @brief Holds the number of size classes: 128, 256, 512, 1024, 2048 and 4096 bytes. */
        inline constexpr std::size_t K_SCRATCH_CLASS_COUNT = 6;

        /** @brief Holds the size class of a block that is not pooled. */
        inline constexpr std::size_t K_SCRATCH_UNPOOLED = K_SCRATCH_CLASS_COUNT;

        /** @brief Holds how many idle blocks one size class of one device keeps. */
        inline constexpr std::size_t K_SCRATCH_IDLE_BLOCKS_PER_CLASS = 64;

        /**
         * @brief Returns the size class that holds bytes, or K_SCRATCH_UNPOOLED above the largest.
         */
        [[nodiscard]] inline constexpr std::size_t scratch_class_of(std::size_t bytes) noexcept
        {
            std::size_t index = 0;
            std::size_t capacity = K_SCRATCH_MIN_BLOCK_BYTES;
            while (capacity < bytes && index < K_SCRATCH_CLASS_COUNT)
            {
                capacity *= 2;
                ++index;
            }
            return index;
        }

        /** @brief Returns the block size of a pooled size class, in bytes. */
        [[nodiscard]] inline constexpr std::size_t scratch_class_bytes(std::size_t size_class) noexcept
        {
            return K_SCRATCH_MIN_BLOCK_BYTES << size_class;
        }

        static_assert(scratch_class_of(1) == 0 && scratch_class_of(129) == 1,
                      "A request is rounded up to the smallest class that holds it.");
        static_assert(scratch_class_bytes(K_SCRATCH_CLASS_COUNT - 1) == K_SCRATCH_MAX_BLOCK_BYTES &&
                      scratch_class_of(K_SCRATCH_MAX_BLOCK_BYTES + 1) == K_SCRATCH_UNPOOLED,
                      "The class table ends at K_SCRATCH_MAX_BLOCK_BYTES.");

        class ScratchShelf;

        /**
         * @struct ScratchBlock
         * @brief Holds one device allocation the shelf lends and the count of handles on it.
         */
        struct ScratchBlock
        {
            /** @brief The device allocation. */
            SushiRuntime::sushi_ptr<Storage> storage;

            /** @brief The device queue that owns the allocation. */
            std::size_t device_index = 0;

            /** @brief The size class, or K_SCRATCH_UNPOOLED. */
            std::size_t size_class = K_SCRATCH_UNPOOLED;

            /** @brief The shelf the block returns to; set only while the block is lent. */
            SushiRuntime::sushi_ptr<ScratchShelf> shelf;

            /** @brief The number of ScratchBuffer handles on the block. */
            std::atomic<std::int32_t> handles{0};
        };

        /**
         * @brief Keeps the idle device scratch blocks of one runtime, per device and size class.
         * @see modules/engine/engine/README.md
         */
        class ScratchShelf final : public SushiRuntime::RefCounted
        {
            public:
                /**
                 * @brief Reserves the idle lists, so returning a block never allocates.
                 * @param runtime Allocates every block; must outlive the shelf.
                 */
                explicit ScratchShelf(SushiRuntime::Runtime& runtime)
                    : runtime_(&runtime), idle_(runtime.advanced().device_count())
                {
                    for (auto& classes : idle_)
                    {
                        for (auto& idle : classes)
                            idle.reserve(K_SCRATCH_IDLE_BLOCKS_PER_CLASS);
                    }
                }

                /**
                 * @brief Lends a device block of at least bytes: an idle one, or a new allocation.
                 * @pre bytes is above zero.
                 * @return A block with no handle on it and no shelf set; the caller owns it.
                 * @throws std::runtime_error if device_index is out of range or the allocation fails.
                 */
                [[nodiscard]] ScratchBlock* take(std::size_t bytes, std::size_t device_index)
                {
                    const std::size_t size_class = scratch_class_of(bytes);

                    if (size_class != K_SCRATCH_UNPOOLED && device_index < idle_.size())
                    {
                        const std::lock_guard<std::mutex> lock(guard_);
                        auto& idle = idle_[device_index][size_class];
                        if (!idle.empty())
                        {
                            ScratchBlock* block = idle.back().release();
                            idle.pop_back();
                            return block;
                        }
                    }

                    const std::size_t capacity = size_class == K_SCRATCH_UNPOOLED
                                                     ? bytes
                                                     : scratch_class_bytes(size_class);

                    auto block = std::make_unique<ScratchBlock>();
                    block->storage = SushiRuntime::make_sushi<Storage>(
                        *runtime_, capacity, SushiRuntime::Residency::Device, device_index);
                    block->device_index = device_index;
                    block->size_class = size_class;
                    return block.release();
                }

                /**
                 * @brief Takes a block back: keeps it idle, or frees it when it is unpooled or its
                 *        class is full.
                 * @pre block came from take() of this shelf, has no handle on it and no shelf set.
                 */
                void give_back(ScratchBlock* block) noexcept
                {
                    std::unique_ptr<ScratchBlock> owned(block);
                    if (owned->size_class == K_SCRATCH_UNPOOLED || owned->device_index >= idle_.size())
                        return;

                    const std::lock_guard<std::mutex> lock(guard_);
                    auto& idle = idle_[owned->device_index][owned->size_class];

                    // The list was reserved to its ceiling, so this push_back cannot allocate.
                    if (idle.size() < K_SCRATCH_IDLE_BLOCKS_PER_CLASS)
                        idle.push_back(std::move(owned));
                }

                /** @brief Returns how many blocks sit idle across every device and class. */
                [[nodiscard]] std::size_t idle_block_count() const
                {
                    const std::lock_guard<std::mutex> lock(guard_);
                    std::size_t count = 0;
                    for (const auto& classes : idle_)
                    {
                        for (const auto& idle : classes)
                            count += idle.size();
                    }
                    return count;
                }

                /** @brief Frees every idle block; lent blocks are not touched. */
                void trim()
                {
                    std::vector<std::unique_ptr<ScratchBlock>> released;
                    const std::lock_guard<std::mutex> lock(guard_);
                    for (auto& classes : idle_)
                    {
                        for (auto& idle : classes)
                        {
                            for (auto& block : idle)
                                released.push_back(std::move(block));
                            idle.clear();
                        }
                    }
                }

            private:
                /// @brief The idle lists of one device, one per size class.
                using ClassLists =
                    std::array<std::vector<std::unique_ptr<ScratchBlock>>, K_SCRATCH_CLASS_COUNT>;

                /// @brief The runtime that allocates every block; never null.
                SushiRuntime::Runtime* runtime_;

                /// @brief Serialises every read and write of idle_.
                mutable std::mutex guard_;

                /// @brief The idle blocks, indexed by device queue and then by size class.
                std::vector<ClassLists> idle_;
        };
    } // namespace Detail
} // namespace SushiBLAS
