/**************************************************************************/
/* late_count_hold.hpp                                                    */
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
 * @file late_count_hold.hpp
 * @brief Declares LateCountHold, which keeps late-count tasks' storages while a recording lives.
 * @author Mustafa Garip
 */

#pragma once

#include <mutex>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    /** @brief Holds what one late-count task needs: the owner token and the count's words. */
    struct LateCount
    {
        /// @brief The token the runtime watches; its deleter holds every storage of the task.
        SushiRuntime::LivenessToken owner;

        /// @brief The whole allocation of the count, borrowed as the runtime's word buffer.
        SushiRuntime::Buffer<std::uint32_t> words;
    };

    /**
     * @brief Keeps the storages of late-count tasks alive for as long as its owner keeps the
     *        recording that holds them.
     *
     * A dynamic node's kernel capture cannot hold storage, so a token stands in for it. The
     * owner of the graph owns the hold; TaskRecorder only records into it. Thread-safe.
     * @see modules/engine/engine/README.md
     */
    class LateCountHold final
    {
        public:
            /**
             * @brief Binds the hold to the runtime its counts are borrowed from.
             * @param runtime Must outlive this hold.
             */
            explicit LateCountHold(SushiRuntime::Runtime& runtime) noexcept
                : runtime_(&runtime)
            {

            }

            LateCountHold(const LateCountHold&) = delete;
            LateCountHold& operator=(const LateCountHold&) = delete;
            LateCountHold(LateCountHold&&) = delete;
            LateCountHold& operator=(LateCountHold&&) = delete;
            /** @brief Releases every kept token, as release() does. */
            ~LateCountHold() = default;

            /**
             * @brief Makes the owner token of a late task and borrows its count under it.
             * @return The token, which holds every storage of operands and of count, and the
             *         count's allocation as words the token owns.
             * @pre count has storage.
             */
            [[nodiscard]] LateCount borrow(const OperandSet& operands,
                                           const Tensor& count) const
            {
                Internal::StorageKeepAlive held = operands.retained();
                held.retain(count.storage);
                LateCount late;
                late.owner = SushiRuntime::LivenessToken(
                    new SushiRuntime::Liveness(),
                    [held = std::move(held)](SushiRuntime::Liveness* life)
                    {
                        delete life;
                    });

                const Storage& storage = *count.storage;
                late.words = runtime_->interop().borrow<std::uint32_t>(
                    static_cast<std::uint32_t*>(storage.allocation_base()),
                    storage.size_bytes() / sizeof(std::uint32_t),
                    SushiRuntime::DeviceIndex{storage.device_index()}, storage.residency(),
                    late.owner);
                return late;
            }

            /** @brief Keeps owner until release(). */
            void keep(SushiRuntime::LivenessToken owner)
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                held_.push_back(std::move(owner));
            }

            /** @brief Drops every kept token, once no recording that needs them will run again. */
            void release() noexcept
            {
                std::vector<SushiRuntime::LivenessToken> dropped;
                {
                    const std::lock_guard<std::mutex> lock(mutex_);
                    dropped.swap(held_);
                }
            }

            /** @brief Returns how many tokens are kept. */
            [[nodiscard]] std::size_t held_count() const
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                return held_.size();
            }

        private:
            /// @brief The runtime counts are borrowed from; never null.
            SushiRuntime::Runtime* runtime_;

            /// @brief Serialises every use of held_.
            mutable std::mutex mutex_;

            /// @brief The tokens kept, one per late task recorded since the last release.
            std::vector<SushiRuntime::LivenessToken> held_;
    };
} // namespace SushiBLAS
