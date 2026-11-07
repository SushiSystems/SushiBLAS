/**************************************************************************/
/* shared_allocator.hpp                                                   */
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
 * @file shared_allocator.hpp
 * @brief Declares SharedAllocator, the port through which SharedArray gets and frees its memory.
 * @author Mustafa Garip
 * @see modules/foundation/host_device/README.md
 */

#pragma once

#include <cstddef>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Gives SharedArray the aligned memory it holds and takes it back. */
        class SharedAllocator
        {
            public:
                SharedAllocator() = default;
                SharedAllocator(const SharedAllocator&) = delete;
                SharedAllocator& operator=(const SharedAllocator&) = delete;
                SharedAllocator(SharedAllocator&&) = delete;
                SharedAllocator& operator=(SharedAllocator&&) = delete;
                virtual ~SharedAllocator() = default;

                /**
                 * @brief Returns @p bytes of memory aligned to @p alignment, or nullptr when there is none.
                 * @param bytes The size of the block.
                 * @param alignment The alignment of its first byte, in bytes.
                 */
                [[nodiscard]] virtual void* allocate(std::size_t bytes, std::size_t alignment) = 0;

                /** @brief Takes back a block this allocator returned and has not taken back yet. */
                virtual void deallocate(void* block) noexcept = 0;
        };
    } // namespace Test
} // namespace SushiBLAS
