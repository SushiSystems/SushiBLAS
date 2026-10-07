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

#include <cstddef>
#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/SushiRuntime.h>

namespace SushiBLAS
{
    /**
     * @brief Holds the byte alignment of every Storage allocation base, which its size is a
     *        multiple of.
     * @see include/SushiBLAS/README.md
     */
    inline constexpr std::size_t K_STORAGE_ALIGNMENT = SushiRuntime::Core::DEFAULT_ALIGNMENT;

    static_assert(K_STORAGE_ALIGNMENT == 128,
                  "Storage documents a 128-byte alignment; SushiRuntime::Core::DEFAULT_ALIGNMENT "
                  "moved, and the documented guarantee has to move with it on purpose.");

    /**
     * @class Storage
     * @brief Owns one reference-counted USM allocation a tensor's elements live in.
     * @see include/SushiBLAS/README.md
     */
    class alignas(64) Storage : public SushiRuntime::Core::RefCounted
    {
        public:
            /**
             * @brief Allocates n_bytes, rounded up to K_STORAGE_ALIGNMENT, at an address aligned to it.
             * @param ctx Runtime context that allocates and frees; must outlive this Storage.
             * @param n_bytes Number of bytes requested; above zero.
             * @param device_index Device queue that owns the allocation.
             * @throws std::runtime_error if the size is zero or overflows, the allocation fails,
             *         or the device returns an address that is not aligned.
             */
            Storage(SushiRuntime::Execution::RuntimeContext& ctx,
                    std::size_t n_bytes,
                    SushiRuntime::API::Residency residency = SushiRuntime::API::Residency::Shared,
                    std::size_t device_index = 0)
                : context_(&ctx), residency_(residency), device_index_(device_index)
            {
                SB_THROW_IF(n_bytes == 0, "Requested size cannot be zero");
                SB_THROW_IF(n_bytes > SIZE_MAX - (K_STORAGE_ALIGNMENT - 1),
                            "Requested size {} overflows when rounded up to {} bytes",
                            n_bytes, K_STORAGE_ALIGNMENT);

                size_bytes_ = (n_bytes + K_STORAGE_ALIGNMENT - 1) / K_STORAGE_ALIGNMENT *
                              K_STORAGE_ALIGNMENT;

                data_ptr_ = allocate_usm(*context_, size_bytes_, residency_, device_index_);
                SB_THROW_IF(!data_ptr_, "Allocation failed for {} bytes", size_bytes_);

                if (reinterpret_cast<std::uintptr_t>(data_ptr_) % K_STORAGE_ALIGNMENT != 0)
                {
                    context_->free_usm(data_ptr_);
                    data_ptr_ = nullptr;
                    SB_THROW_IF(true, "The device returned an allocation that is not {}-byte aligned",
                                K_STORAGE_ALIGNMENT);
                }
            }

            Storage(const Storage&) = delete;
            Storage& operator=(const Storage&) = delete;

            /**
             * @brief Returns the base address of the whole allocation, ignoring every view's offset.
             * @return An address that is a multiple of K_STORAGE_ALIGNMENT.
             * @warning Never a tensor's data; use Tensor::data() to address elements.
             * @see include/SushiBLAS/README.md
             */
            void* allocation_base() const noexcept { return data_ptr_; }

            /**
             * @brief Returns the bytes allocated: the requested size rounded up to K_STORAGE_ALIGNMENT.
             */
            [[nodiscard]] std::size_t size_bytes() const noexcept { return size_bytes_; }

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
            [[nodiscard]] std::size_t get_device_index() const noexcept { return device_index_; }

        protected:
            /** @brief Frees the allocation through the context that made it. */
            ~Storage() override
            {
                if (data_ptr_)
                {
                    context_->free_usm(data_ptr_);
                }
            }

        private:
            /**
             * @brief Allocates USM of one residency, aligned to K_STORAGE_ALIGNMENT, on one device queue.
             * @return The USM pointer, or nullptr on failure.
             */
            static void* allocate_usm(SushiRuntime::Execution::RuntimeContext& ctx, std::size_t bytes,
                                      SushiRuntime::API::Residency residency, std::size_t device_index)
            {
                switch (residency)
                {
                    case SushiRuntime::API::Residency::Device:
                        return ctx.aligned_malloc_device_on(K_STORAGE_ALIGNMENT, bytes, device_index);
                    case SushiRuntime::API::Residency::Host:
                        return ctx.aligned_malloc_host_on(K_STORAGE_ALIGNMENT, bytes, device_index);
                    case SushiRuntime::API::Residency::Shared:
                        return ctx.aligned_malloc_shared_on(K_STORAGE_ALIGNMENT, bytes, device_index);
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
            std::size_t device_index_ = 0;

            /** @brief The bytes allocated, a multiple of K_STORAGE_ALIGNMENT. */
            std::size_t size_bytes_ = 0;
    };
} // namespace SushiBLAS
