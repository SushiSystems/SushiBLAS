/****************************************************************/
/* storage.hpp                                                  */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file storage.hpp
 * @brief Declares Storage, the reference-counted USM allocation a tensor's elements live in.
 * @author Mustafa Garip
 */

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
             * @brief Allocates aligned memory of n_bytes through the allocator, under a strategy.
             * @throws std::runtime_error if the allocator is null or the allocation fails.
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
             * @brief Returns the base address of the whole allocation, ignoring every view's offset.
             * @return The pointer the allocator handed back, or nullptr before construction completed.
             * @warning Never a tensor's data; use Tensor::data() to address elements.
             * @see include/SushiBLAS/README.md
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
