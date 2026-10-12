/**************************************************************************/
/* operand_set.hpp                                                        */
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
 * @file operand_set.hpp
 * @brief Declares OperandSet, the tensors one task reads and writes, and the storage it retains.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <vector>
#include <cstddef>
#include <algorithm>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        /**
         * @brief Holds one reference to each distinct Storage a recorded task must keep allocated.
         * @see modules/engine/engine/README.md
         */
        class StorageKeepAlive final
        {
            public:
                /** @brief Holds how many storages fit before the list spills to the heap. */
                static constexpr std::size_t K_INLINE_CAPACITY = 6;

                /** @brief Takes a reference to storage unless it is null or already held. */
                void retain(const SushiRuntime::sushi_ptr<Storage>& storage)
                {
                    if (storage == nullptr || holds(storage.get()))
                        return;

                    if (count_ < K_INLINE_CAPACITY)
                        inline_[count_] = storage;
                    else
                        overflow_.push_back(storage);
                    ++count_;
                }

                /** @brief Takes a reference to every storage other holds. */
                void retain_all(const StorageKeepAlive& other)
                {
                    const std::size_t inline_count = std::min(other.count_, K_INLINE_CAPACITY);
                    for (std::size_t i = 0; i < inline_count; ++i)
                        retain(other.inline_[i]);
                    for (const SushiRuntime::sushi_ptr<Storage>& held : other.overflow_)
                        retain(held);
                }

                /** @brief Returns how many distinct storages are held. */
                [[nodiscard]] std::size_t size() const noexcept { return count_; }

            private:
                /** @brief Reports whether storage is already held. */
                [[nodiscard]] bool holds(const Storage* storage) const noexcept
                {
                    const std::size_t inline_count = std::min(count_, K_INLINE_CAPACITY);
                    for (std::size_t i = 0; i < inline_count; ++i)
                    {
                        if (inline_[i].get() == storage)
                            return true;
                    }
                    for (const SushiRuntime::sushi_ptr<Storage>& held : overflow_)
                    {
                        if (held.get() == storage)
                            return true;
                    }
                    return false;
                }

                /// @brief The first K_INLINE_CAPACITY storages, in the order they were retained.
                std::array<SushiRuntime::sushi_ptr<Storage>, K_INLINE_CAPACITY> inline_{};

                /// @brief The storages past the inline capacity.
                std::vector<SushiRuntime::sushi_ptr<Storage>> overflow_;

                /// @brief How many storages inline_ and overflow_ hold together.
                std::size_t count_ = 0;
        };
    } // namespace Internal

    /**
     * @brief Collects the tensors one task touches: their regions and the storage they live in.
     * @see modules/engine/engine/README.md
     */
    class OperandSet final
    {
        public:
            /** @brief Declares that the task reads operand; a tensor without storage adds nothing. */
            OperandSet& reads(const Tensor& operand)
            {
                if (!operand.storage)
                    return *this;

                append(reads_, operand.region());
                retain(operand);
                return *this;
            }

            /** @brief Declares that the task writes operand; a tensor without storage adds nothing. */
            OperandSet& writes(const Tensor& operand)
            {
                if (!operand.storage)
                    return *this;

                append(writes_, operand.region());
                retain(operand);
                return *this;
            }

            /** @brief Declares that the task reads and writes operand in place. */
            OperandSet& updates(const Tensor& operand)
            {
                return reads(operand).writes(operand);
            }

            /** @brief Declares that the task writes a device scratch; an empty one adds nothing. */
            OperandSet& scratch(const ScratchBuffer& buffer)
            {
                if (buffer.is_empty())
                    return *this;

                append(writes_, buffer.region());
                retained_.retain(buffer.storage());
                return *this;
            }

            /** @brief Adds every declaration of other to this set. */
            OperandSet& merge(const OperandSet& other)
            {
                for (const SushiRuntime::BufferRegion& region : other.reads_)
                    append(reads_, region);
                for (const SushiRuntime::BufferRegion& region : other.writes_)
                    append(writes_, region);
                retained_.retain_all(other.retained_);
                has_device_only_operand_ = has_device_only_operand_ || other.has_device_only_operand_;
                return *this;
            }

            /** @brief Returns the byte intervals the task reads, in declaration order. */
            [[nodiscard]] const std::vector<SushiRuntime::BufferRegion>&
            read_regions() const noexcept
            {
                return reads_;
            }

            /** @brief Returns the byte intervals the task writes, in declaration order. */
            [[nodiscard]] const std::vector<SushiRuntime::BufferRegion>&
            write_regions() const noexcept
            {
                return writes_;
            }

            /** @brief Returns the storages the recorded task holds a reference to. */
            [[nodiscard]] const Internal::StorageKeepAlive& retained() const noexcept
            {
                return retained_;
            }

            /** @brief Reports whether a declared tensor lives in device-only USM. */
            [[nodiscard]] bool has_device_only_operand() const noexcept
            {
                return has_device_only_operand_;
            }

        private:
            /** @brief Holds the region count reserved on the first declaration. */
            static constexpr std::size_t K_TYPICAL_REGIONS = 4;

            /** @brief Appends region, reserving K_TYPICAL_REGIONS on the first call. */
            static void append(std::vector<SushiRuntime::BufferRegion>& regions,
                               const SushiRuntime::BufferRegion& region)
            {
                if (regions.empty())
                    regions.reserve(K_TYPICAL_REGIONS);
                regions.push_back(region);
            }

            /**
             * @brief Retains the storage of operand and notes a device-only residency.
             * @pre operand has storage.
             */
            void retain(const Tensor& operand)
            {
                retained_.retain(operand.storage);
                if (operand.storage->residency() == SushiRuntime::Residency::Device)
                    has_device_only_operand_ = true;
            }

            /// @brief The regions declared through reads() and updates().
            std::vector<SushiRuntime::BufferRegion> reads_;

            /// @brief The regions declared through writes(), updates() and scratch().
            std::vector<SushiRuntime::BufferRegion> writes_;

            /// @brief One reference per distinct storage behind the declared regions.
            Internal::StorageKeepAlive retained_;

            /// @brief Whether a declared tensor is Residency::Device.
            bool has_device_only_operand_ = false;
    };
} // namespace SushiBLAS
