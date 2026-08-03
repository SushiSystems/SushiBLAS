/**************************************************************************/
/* storage.hpp                                                            */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

#include <cassert>
#include <cstddef>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/SushiRuntime.h>

namespace SushiBLAS 
{
    /**
     * @class Storage
     * @brief A class to manage memory for tensors.
     * 
     * This class holds the pointer to memory and information about the allocation.
     * it uses SYCL Unified Shared Memory (USM) through a SushiRuntime allocator.
     * It is reference counted to help share data safely between tensors.
     */
    class alignas(64) Storage : public SushiRuntime::Core::RefCounted
    {
        public:
            /** @brief The total number of bytes allocated in memory. */
            size_t size_bytes = 0;         
            
            /** @brief The number of bytes that the user actually asked for. */
            size_t requested_bytes = 0;    

            /** @brief The allocator used to manage this memory. */
            SushiRuntime::sushi_ptr<SushiRuntime::Memory::USMAllocator> allocator; 
            
            /** @brief The strategy used for memory allocation (Shared, Device, or Host singleton). */
            const SushiRuntime::Memory::IAllocatorStrategy* strategy = nullptr;

            /**
            * @brief Create a new Storage object.
            *
            * This allocates memory using the provided allocator and strategy.
            * The size is automatically aligned for better performance.
            *
            * @param alloc The allocator to use.
            * @param n_bytes The amount of memory to allocate in bytes.
            * @param strat The allocation strategy to use (default is Shared).
            * @throws std::runtime_error If the allocator is null or allocation fails.
            */
            Storage(SushiRuntime::sushi_ptr<SushiRuntime::Memory::USMAllocator> alloc,
                    size_t n_bytes,
                    const SushiRuntime::Memory::IAllocatorStrategy& strat = SushiRuntime::Memory::Strategies::Shared::instance())
                : requested_bytes(n_bytes), allocator(alloc), strategy(&strat)
            {
                SB_THROW_IF(!allocator, "Allocator pointer cannot be null");
                SB_THROW_IF(n_bytes == 0, "Requested size cannot be zero");

                size_bytes = n_bytes;
                // Align to modern HPC alignment
                if (size_bytes % SushiRuntime::Core::DEFAULT_ALIGNMENT != 0) 
                {
                    size_bytes += SushiRuntime::Core::DEFAULT_ALIGNMENT - (size_bytes % SushiRuntime::Core::DEFAULT_ALIGNMENT);
                }

                data_ptr_ = allocator->allocate(size_bytes, *strategy);
                SB_THROW_IF(!data_ptr_, "Allocation failed for {} bytes", size_bytes);
            }

            Storage(const Storage&) = delete;
            Storage& operator=(const Storage&) = delete;

            /**
             * @brief Base address of the whole allocation, ignoring every view's offset.
             *
             * @return The pointer the allocator handed back, or nullptr before construction
             *         completed.
             *
             * @warning This is the *allocation* base, never a tensor's data. Handing this
             *          pointer to a kernel silently ignores `Tensor::storage_offset`, so a
             *          tensor that is a view at a non-zero offset reads or writes the start
             *          of the allocation instead of its own elements - which is exactly the
             *          defect that made this member private. Use `Tensor::data()` /
             *          `Tensor::data_as<T>()` to address a tensor's elements, and
             *          `Tensor::region()` to describe them to the dependency tracker. The
             *          only legitimate uses of the base are identity ("which allocation is
             *          this?") and deallocation.
             */
            void* allocation_base() const noexcept { return data_ptr_; }

        protected:
            /**
             * @brief Destroy the Storage object.
             *
             * This will automatically free the allocated memory.
             */
            ~Storage() override
            {
                if (data_ptr_ && allocator)
                {
                    allocator->deallocate(data_ptr_);
                }
            }

        private:
            /**
             * @brief The pointer to the raw data in memory.
             *
             * Private, and reachable only through @ref allocation_base(), because the whole
             * BLAS layer used to reach for it directly and thereby drop `storage_offset` on
             * the floor. Making the wrong pointer un-spellable is what stops the next
             * operation added to the library from reintroducing that.
             */
            void* data_ptr_ = nullptr;
    };
} // namespace SushiBLAS
