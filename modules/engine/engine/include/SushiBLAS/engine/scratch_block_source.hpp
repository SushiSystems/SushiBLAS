/**************************************************************************/
/* scratch_block_source.hpp                                               */
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
 * @file scratch_block_source.hpp
 * @brief Declares ScratchBlockSource, the port a ScratchPool takes its blocks from.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>

namespace SushiBLAS
{
    /**
     * @brief Serves the allocations a ScratchPool lends as scratch blocks.
     *
     * RuntimeScratchSource allocates device memory from a SushiRuntime::Runtime.
     * HostScratchSource serves host memory and counts what it served.
     * @see modules/engine/engine/README.md
     */
    class ScratchBlockSource
    {
        public:
            ScratchBlockSource(const ScratchBlockSource&) = delete;
            ScratchBlockSource& operator=(const ScratchBlockSource&) = delete;
            ScratchBlockSource(ScratchBlockSource&&) = delete;
            ScratchBlockSource& operator=(ScratchBlockSource&&) = delete;
            virtual ~ScratchBlockSource() = default;

            /** @brief Returns how many device queues the source serves blocks for; at least 1. */
            [[nodiscard]] virtual std::size_t device_count() const = 0;

            /**
             * @brief Serves one allocation of at least @p bytes for a device queue.
             * @param bytes Size requested; above zero.
             * @param device_index A device of this source, below device_count().
             * @return A storage no other live block shares an address with; never null.
             * @throws std::runtime_error if bytes is zero, device_index is out of range or the
             *         allocation fails.
             */
            [[nodiscard]] virtual SushiRuntime::sushi_ptr<Storage> serve(
                std::size_t bytes, std::size_t device_index) = 0;

        protected:
            /** @brief Lets an adapter construct itself; the port is never built alone. */
            ScratchBlockSource() = default;
    };
} // namespace SushiBLAS
