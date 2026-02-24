/**************************************************************************/
/* storage.hpp                                                            */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                   	  */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
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
            /** @brief The pointer to the raw data in memory. */
            void* data_ptr = nullptr;      
            
            /** @brief The total number of bytes allocated in memory. */
            size_t size_bytes = 0;         
            
            /** @brief The number of bytes that the user actually asked for. */
            size_t requested_bytes = 0;    

            /** @brief The allocator used to manage this memory. */
            SushiRuntime::sushi_ptr<SushiRuntime::Memory::USMAllocator> allocator; 
            
            /** @brief The strategy used for memory allocation (Shared, Device, or Host). */
            SushiRuntime::Memory::AllocStrategy strategy;                          

            /**
            * @brief Create a new Storage object.
            * 
            * This allocates memory using the provided allocator and strategy.
            * The size is automatically aligned for better performance.
            * 
            * @param alloc The allocator to use.
            * @param n_bytes The amount of memory to allocate in bytes.
            * @param strat The allocation strategy to use (default is SHARED).
            * @throws std::runtime_error If the allocator is null or allocation fails.
            */
            Storage(SushiRuntime::sushi_ptr<SushiRuntime::Memory::USMAllocator> alloc, 
                    size_t n_bytes, 
                    SushiRuntime::Memory::AllocStrategy strat = SushiRuntime::Memory::AllocStrategy::SHARED) 
                : requested_bytes(n_bytes), allocator(alloc), strategy(strat)
            {
                SB_THROW_IF(!allocator, "Allocator pointer cannot be null");
                SB_THROW_IF(n_bytes == 0, "Requested size cannot be zero");

                size_bytes = n_bytes;
                // Align to modern HPC alignment
                if (size_bytes % SushiRuntime::Core::DEFAULT_ALIGNMENT != 0) 
                {
                    size_bytes += SushiRuntime::Core::DEFAULT_ALIGNMENT - (size_bytes % SushiRuntime::Core::DEFAULT_ALIGNMENT);
                }

                data_ptr = allocator->allocate(size_bytes, strategy);
                SB_THROW_IF(!data_ptr, "Allocation failed for {} bytes", size_bytes);
            }

            Storage(const Storage&) = delete;
            Storage& operator=(const Storage&) = delete;

        protected:
            /** 
             * @brief Destroy the Storage object.
             * 
             * This will automatically free the allocated memory.
             */
            ~Storage() override 
            {
                if (data_ptr && allocator) 
                {
                    allocator->deallocate(data_ptr);
                }
            }
    };
} // namespace SushiBLAS