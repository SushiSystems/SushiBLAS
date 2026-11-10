/**************************************************************************/
/* host_scratch_source.hpp                                                */
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
 * @file host_scratch_source.hpp
 * @brief Declares HostScratchSource, the ScratchBlockSource that serves host memory and counts it served and released.
 * @author Mustafa Garip
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <utility>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/engine/scratch_block_source.hpp>
#include <SushiBLAS/tensor/host_block.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/storage_block.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>

namespace SushiBLAS
{
    /**
     * @brief Serves every block as a host Storage no runtime owns, for one device queue.
     *
     * It needs no runtime, so a ScratchPool over it runs in a test that makes no device. Its
     * counters are atomic, so the pool may be used from several threads. They count blocks served
     * and blocks released, so a test reads how many allocations are still alive.
     * @see modules/engine/engine/README.md
     */
    class HostScratchSource final : public ScratchBlockSource
    {
        public:
            /** @brief Builds a source that has served nothing. */
            HostScratchSource() noexcept = default;

            /** @brief Returns 1: the host has one queue's worth of blocks. */
            [[nodiscard]] std::size_t device_count() const override { return 1; }

            /**
             * @brief Allocates a host Storage of at least @p bytes and counts it.
             * @throws std::runtime_error if bytes is zero or device_index is not 0.
             * @throws std::bad_alloc if the allocation fails.
             */
            [[nodiscard]] SushiRuntime::sushi_ptr<Storage> serve(
                std::size_t bytes, std::size_t device_index) override
            {
                SB_THROW_IF(device_index >= device_count(),
                            "Device index {} is out of range; the host source serves {} device",
                            device_index, device_count());

                return SushiRuntime::make_sushi<Storage>(
                    std::make_unique<CountedBlock>(bytes, counters_));
            }

            /** @brief Returns how many blocks the source served since it was made. */
            [[nodiscard]] std::size_t served_blocks() const noexcept
            {
                return counters_->served_blocks.load(std::memory_order_relaxed);
            }

            /** @brief Returns the sum of size_bytes() over the blocks served since it was made. */
            [[nodiscard]] std::size_t served_bytes() const noexcept
            {
                return counters_->served_bytes.load(std::memory_order_relaxed);
            }

            /** @brief Returns how many served blocks were released, because the last Storage holding each was dropped. */
            [[nodiscard]] std::size_t released_blocks() const noexcept
            {
                return counters_->released_blocks.load(std::memory_order_relaxed);
            }

            /** @brief Returns how many served blocks are still alive: served minus released. */
            [[nodiscard]] std::size_t live_blocks() const noexcept
            {
                return served_blocks() - released_blocks();
            }

        private:
            /// @brief Holds the counters, which outlive the source while a block it served lives.
            struct Counters
            {
                /// @brief How many blocks were served.
                std::atomic<std::size_t> served_blocks{0};

                /// @brief The bytes of the blocks served.
                std::atomic<std::size_t> served_bytes{0};

                /// @brief How many served blocks were released.
                std::atomic<std::size_t> released_blocks{0};
            };

            /// @brief A HostBlock that counts itself served when it is made and released when it dies.
            class CountedBlock final : public StorageBlock
            {
                public:
                    /** @brief Allocates bytes of host memory aligned to K_STORAGE_ALIGNMENT and counts it served. */
                    CountedBlock(std::size_t bytes, std::shared_ptr<Counters> counters)
                        : block_(rounded_storage_size(bytes), K_STORAGE_ALIGNMENT),
                          counters_(std::move(counters))
                    {
                        counters_->served_blocks.fetch_add(1, std::memory_order_relaxed);
                        counters_->served_bytes.fetch_add(block_.size_bytes(),
                                                          std::memory_order_relaxed);
                    }

                    /** @brief Counts the block released; the memory is freed with block_. */
                    ~CountedBlock() override
                    {
                        counters_->released_blocks.fetch_add(1, std::memory_order_relaxed);
                    }

                    /** @brief Returns the first byte. */
                    [[nodiscard]] void* data() const noexcept override { return block_.data(); }

                    /** @brief Returns the bytes owned. */
                    [[nodiscard]] std::size_t size_bytes() const noexcept override
                    {
                        return block_.size_bytes();
                    }

                    /** @brief Returns Residency::Host. */
                    [[nodiscard]] SushiRuntime::Residency residency() const noexcept override
                    {
                        return block_.residency();
                    }

                    /** @brief Returns 0. */
                    [[nodiscard]] std::size_t device_index() const noexcept override
                    {
                        return block_.device_index();
                    }

                    /** @brief Returns nullptr. */
                    [[nodiscard]] sycl::queue* queue() const override { return block_.queue(); }

                    /** @brief Copies bytes out of the block. */
                    void copy_to_host(void* destination, std::size_t offset,
                                      std::size_t bytes) const override
                    {
                        block_.copy_to_host(destination, offset, bytes);
                    }

                    /** @brief Copies bytes into the block. */
                    void copy_from_host(const void* source, std::size_t offset,
                                        std::size_t bytes) const override
                    {
                        block_.copy_from_host(source, offset, bytes);
                    }

                private:
                    /// @brief The memory.
                    HostBlock block_;

                    /// @brief The counters of the source that served the block.
                    std::shared_ptr<Counters> counters_;
            };

            /// @brief The counters; shared with every block served so a block may outlive the source.
            std::shared_ptr<Counters> counters_ = std::make_shared<Counters>();
    };
} // namespace SushiBLAS
