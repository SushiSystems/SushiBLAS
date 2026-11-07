/**************************************************************************/
/* block_maker.hpp                                                        */
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
 * @file block_maker.hpp
 * @brief Declares BlockMaker, the port a TensorFactory gets its StorageBlocks through.
 * @author Mustafa Garip
 * @see modules/data/tensor/README.md
 */

#pragma once

#include <cstddef>
#include <memory>
#include <SushiBLAS/tensor/storage_block.hpp>

namespace SushiBLAS
{
    /**
     * @class BlockMaker
     * @brief Makes the StorageBlocks of one kind of memory: host memory or a runtime's buffers.
     * @see modules/data/tensor/README.md
     */
    class BlockMaker
    {
        public:
            virtual ~BlockMaker() = default;

            /**
             * @brief Allocates a block of n_bytes rounded up to K_STORAGE_ALIGNMENT.
             * @param n_bytes Number of bytes requested; above zero.
             * @param residency Where the memory lives; a maker of host memory ignores it.
             * @param device_index Device queue that owns the allocation; a device of the maker.
             * @throws std::runtime_error if the size is zero or overflows, the device index is out
             *         of range, or the allocation fails.
             */
            [[nodiscard]] virtual std::unique_ptr<StorageBlock> make(
                std::size_t n_bytes, SushiRuntime::Residency residency,
                std::size_t device_index) const = 0;

            /** @brief Returns the number of device queues a block can sit on; 1 for host memory. */
            [[nodiscard]] virtual std::size_t device_count() const = 0;
    };
} // namespace SushiBLAS
