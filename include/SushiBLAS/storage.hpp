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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

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
     * it uses SYCL Unified Shared Memory (USM) through a SushiRuntime RuntimeContext.
     * It is reference counted to help share data safely between tensors.
     */
    class alignas(64) Storage : public SushiRuntime::Core::RefCounted
    {
        public:
            /** @brief The total number of bytes allocated in memory. */
            size_t size_bytes = 0;         
            
            /** @brief The number of bytes that the user actually asked for. */
            size_t requested_bytes = 0;

            /**
             * @brief Allocates aligned memory of n_bytes from a runtime context, with a residency.
             * @param ctx          Runtime context that performs the allocation; must outlive this Storage.
             * @param n_bytes      Number of bytes requested.
             * @param residency    Kind of USM to allocate.
             * @param device_index Device queue that owns the allocation.
             * @throws std::runtime_error if the size is zero or the allocation fails.
             */
            Storage(SushiRuntime::Execution::RuntimeContext& ctx,
                    size_t n_bytes,
                    SushiRuntime::API::Residency residency = SushiRuntime::API::Residency::Shared,
                    size_t device_index = 0)
                : requested_bytes(n_bytes), context_(&ctx), residency_(residency), device_index_(device_index)
            {
                SB_THROW_IF(n_bytes == 0, "Requested size cannot be zero");

                size_bytes = n_bytes;
                // Align to modern HPC alignment
                if (size_bytes % SushiRuntime::Core::DEFAULT_ALIGNMENT != 0) 
                {
                    size_bytes += SushiRuntime::Core::DEFAULT_ALIGNMENT - (size_bytes % SushiRuntime::Core::DEFAULT_ALIGNMENT);
                }

                data_ptr_ = allocate_usm(*context_, size_bytes, residency_, device_index_);
                SB_THROW_IF(!data_ptr_, "Allocation failed for {} bytes", size_bytes);
            }

            Storage(const Storage&) = delete;
            Storage& operator=(const Storage&) = delete;

            /**
             * @brief Returns the base address of the whole allocation, ignoring every view's offset.
             * @return The pointer the runtime context handed back, or nullptr before construction completed.
             * @warning Never a tensor's data; use Tensor::data() to address elements.
             * @see include/SushiBLAS/README.md
             */
            void* allocation_base() const noexcept { return data_ptr_; }

            /**
             * @brief Returns the runtime context that made the allocation and frees it.
             * @warning Non-owning; the context must outlive every Storage, as it must outlive Engine.
             */
            [[nodiscard]] SushiRuntime::Execution::RuntimeContext& get_context() const noexcept
            {
                return *context_;
            }

            /** @brief Returns the kind of USM the allocation lives in (Shared, Device, or Host). */
            [[nodiscard]] SushiRuntime::API::Residency get_residency() const noexcept
            {
                return residency_;
            }

            /** @brief Returns the device queue that owns the allocation. */
            [[nodiscard]] size_t get_device_index() const noexcept { return device_index_; }

        protected:
            /**
             * @brief Destroy the Storage object.
             *
             * This will automatically free the allocated memory.
             */
            ~Storage() override
            {
                if (data_ptr_)
                {
                    context_->free_usm(data_ptr_);
                }
            }

        private:
            /**
             * @brief Allocates USM of one residency on one device queue.
             * @param ctx          Runtime context that performs the allocation.
             * @param bytes        Allocation size in bytes.
             * @param residency    Kind of USM to allocate.
             * @param device_index Device queue that owns the allocation.
             * @return The USM pointer, or nullptr on failure.
             */
            static void* allocate_usm(SushiRuntime::Execution::RuntimeContext& ctx, size_t bytes,
                                      SushiRuntime::API::Residency residency, size_t device_index)
            {
                switch (residency)
                {
                    case SushiRuntime::API::Residency::Device: return ctx.malloc_device_on(bytes, device_index);
                    case SushiRuntime::API::Residency::Host:   return ctx.malloc_host_on(bytes, device_index);
                    case SushiRuntime::API::Residency::Shared: return ctx.malloc_shared_on(bytes, device_index);
                }
                return nullptr;
            }

            /** @brief The base address of the allocation, read through allocation_base(). */
            void* data_ptr_ = nullptr;

            /** @brief The runtime context that made the allocation; never null. */
            SushiRuntime::Execution::RuntimeContext* context_ = nullptr;

            /** @brief The kind of USM the allocation lives in. */
            SushiRuntime::API::Residency residency_ = SushiRuntime::API::Residency::Shared;

            /** @brief The device queue that owns the allocation. */
            size_t device_index_ = 0;
    };
} // namespace SushiBLAS
