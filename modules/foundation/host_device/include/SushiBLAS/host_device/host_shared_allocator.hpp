/**************************************************************************/
/* host_shared_allocator.hpp                                              */
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
 * @file host_shared_allocator.hpp
 * @brief Declares HostSharedAllocator, the SharedAllocator that serves blocks from host memory and counts them.
 * @author Mustafa Garip
 * @see modules/foundation/host_device/README.md
 */

#pragma once

#include <cstddef>
#include <cstring>
#include <map>
#include <new>

#include <SushiBLAS/host_device/shared_allocator.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds the byte HostSharedAllocator writes over every block it serves. */
        inline constexpr unsigned char K_HOST_BLOCK_POISON = 0xCD;

        /**
         * @brief Serves aligned host memory and counts what it served, with no device and no queue.
         *
         * Every block starts filled with K_HOST_BLOCK_POISON, so a reader of memory nobody wrote sees it.
         * A test reads how many blocks went out and came back, the size and alignment of the last
         * request, and can make the requests fail.
         */
        class HostSharedAllocator final : public SharedAllocator
        {
            public:
                HostSharedAllocator() = default;

                /** @brief Frees the blocks nobody took back. */
                ~HostSharedAllocator() override
                {
                    for (const auto& [block, alignment] : blocks_)
                        ::operator delete(block, std::align_val_t{alignment});
                }

                /** @brief Makes every later request return nullptr when @p fail is true. */
                void set_failing(bool fail) noexcept
                {
                    failing_ = fail;
                }

                /** @brief Returns a block of @p bytes aligned to @p alignment, or nullptr while failing. */
                [[nodiscard]] void* allocate(std::size_t bytes, std::size_t alignment) override
                {
                    last_bytes_ = bytes;
                    last_alignment_ = alignment;
                    if (failing_)
                        return nullptr;
                    void* block = ::operator new(bytes, std::align_val_t{alignment}, std::nothrow);
                    if (block == nullptr)
                        return nullptr;
                    std::memset(block, K_HOST_BLOCK_POISON, bytes);
                    blocks_.emplace(block, alignment);
                    ++allocations_;
                    return block;
                }

                /** @brief Frees a block this allocator returned. */
                void deallocate(void* block) noexcept override
                {
                    const auto found = blocks_.find(block);
                    if (found == blocks_.end())
                        return;
                    ::operator delete(block, std::align_val_t{found->second});
                    blocks_.erase(found);
                    ++frees_;
                }

                /** @brief Returns how many blocks went out. */
                [[nodiscard]] std::size_t allocation_count() const noexcept
                {
                    return allocations_;
                }

                /** @brief Returns how many blocks came back. */
                [[nodiscard]] std::size_t free_count() const noexcept
                {
                    return frees_;
                }

                /** @brief Returns the byte count of the last request, served or not. */
                [[nodiscard]] std::size_t last_bytes() const noexcept
                {
                    return last_bytes_;
                }

                /** @brief Returns the alignment of the last request, served or not. */
                [[nodiscard]] std::size_t last_alignment() const noexcept
                {
                    return last_alignment_;
                }

            private:
                /// @brief The blocks out, with the alignment each was made at.
                std::map<void*, std::size_t> blocks_;
                /// @brief How many blocks went out.
                std::size_t allocations_ = 0;
                /// @brief How many blocks came back.
                std::size_t frees_ = 0;
                /// @brief The byte count of the last request.
                std::size_t last_bytes_ = 0;
                /// @brief The alignment of the last request.
                std::size_t last_alignment_ = 0;
                /// @brief Whether a request returns nullptr.
                bool failing_ = false;
        };
    } // namespace Test
} // namespace SushiBLAS
