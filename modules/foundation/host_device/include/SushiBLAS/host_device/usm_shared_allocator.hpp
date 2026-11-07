/**************************************************************************/
/* usm_shared_allocator.hpp                                               */
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
 * @file usm_shared_allocator.hpp
 * @brief Declares UsmSharedAllocator, the SharedAllocator that allocates shared USM on a queue.
 * @author Mustafa Garip
 * @see modules/foundation/host_device/README.md
 */

#pragma once

#include <cstddef>
#include <sycl/sycl.hpp>
#include <SushiBLAS/host_device/shared_allocator.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Allocates shared USM in the context of one queue, where host code and a kernel both read it. */
        class UsmSharedAllocator final : public SharedAllocator
        {
            public:
                /** @brief Allocates on the device of @p queue, which must outlive this allocator. */
                explicit UsmSharedAllocator(sycl::queue& queue) noexcept
                    : queue_(queue)
                {

                }

                /** @brief Returns shared USM of @p bytes aligned to @p alignment, or nullptr. */
                [[nodiscard]] void* allocate(std::size_t bytes, std::size_t alignment) override
                {
                    return sycl::aligned_alloc_shared(alignment, bytes, queue_);
                }

                /** @brief Frees a block this allocator returned. */
                void deallocate(void* block) noexcept override
                {
                    sycl::free(block, queue_);
                }

            private:
                /// @brief The queue whose context owns the blocks.
                sycl::queue& queue_;
        };
    } // namespace Test
} // namespace SushiBLAS
