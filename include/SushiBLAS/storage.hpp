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

#include <cstddef>
#include <cassert>

#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/SushiRuntime.h>

namespace SushiBLAS 
{
    /**
     * @brief Underlying data storage for tensors.
     */
    class alignas(64) Storage : public SushiRuntime::Core::RefCounted 
    {
    public:
        void* data_ptr = nullptr;
        size_t size_bytes = 0;       
        size_t requested_bytes = 0;

        SushiRuntime::sushi_ptr<SushiRuntime::Memory::USMAllocator> allocator;
        SushiRuntime::Memory::AllocStrategy strategy;

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
        ~Storage() override 
        {
            if (data_ptr && allocator) 
            {
                allocator->deallocate(data_ptr);
            }
        }
    };
} // namespace SushiBLAS