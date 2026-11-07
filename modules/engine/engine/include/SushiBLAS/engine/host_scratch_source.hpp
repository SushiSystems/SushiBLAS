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
 * @brief Declares HostScratchSource, the ScratchBlockSource that serves host memory and counts it.
 * @author Mustafa Garip
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/engine/scratch_block_source.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>

namespace SushiBLAS
{
    /**
     * @brief Serves every block as a host Storage no runtime owns, for one device queue.
     *
     * It needs no runtime, so a ScratchPool over it runs in a test that makes no device. Its
     * counters are atomic, so the pool may be used from several threads.
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

                SushiRuntime::sushi_ptr<Storage> block =
                    SushiRuntime::make_sushi<Storage>(Storage::Host{}, bytes);
                served_blocks_.fetch_add(1, std::memory_order_relaxed);
                served_bytes_.fetch_add(block->size_bytes(), std::memory_order_relaxed);
                return block;
            }

            /** @brief Returns how many blocks the source served since it was made. */
            [[nodiscard]] std::size_t served_blocks() const noexcept
            {
                return served_blocks_.load(std::memory_order_relaxed);
            }

            /** @brief Returns the sum of size_bytes() over the blocks served since it was made. */
            [[nodiscard]] std::size_t served_bytes() const noexcept
            {
                return served_bytes_.load(std::memory_order_relaxed);
            }

        private:
            /// @brief How many blocks were served.
            std::atomic<std::size_t> served_blocks_{0};

            /// @brief The bytes of the blocks served.
            std::atomic<std::size_t> served_bytes_{0};
    };
} // namespace SushiBLAS
