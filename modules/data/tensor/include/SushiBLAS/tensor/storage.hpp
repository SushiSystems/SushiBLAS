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
#include <utility>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    /**
     * @brief Holds the byte alignment of every Storage allocation base, which its size is a
     *        multiple of.
     * @see modules/README.md
     */
    inline constexpr std::size_t K_STORAGE_ALIGNMENT = SushiRuntime::K_DEFAULT_ALIGNMENT;

    static_assert(K_STORAGE_ALIGNMENT == 128,
                  "Storage documents a 128-byte alignment; SushiRuntime::K_DEFAULT_ALIGNMENT "
                  "moved, and the documented guarantee has to move with it on purpose.");

    /**
     * @class Storage
     * @brief Owns one reference-counted USM allocation a tensor's elements live in.
     * @see modules/README.md
     */
    class alignas(64) Storage : public SushiRuntime::RefCounted
    {
        public:
            /**
             * @brief Allocates n_bytes, rounded up to K_STORAGE_ALIGNMENT, at an address aligned to it.
             * @param runtime Allocates the buffer this Storage owns.
             * @param n_bytes Number of bytes requested; above zero.
             * @param device_index Device queue that owns the allocation; a device of runtime.
             * @throws std::runtime_error if the size is zero or overflows, the device index is out
             *         of range, the allocation fails, or the address is not aligned.
             */
            Storage(SushiRuntime::Runtime& runtime,
                    std::size_t n_bytes,
                    SushiRuntime::Residency residency = SushiRuntime::Residency::Shared,
                    std::size_t device_index = 0)
                : controls_(runtime.advanced())
            {
                SB_THROW_IF(n_bytes == 0, "Requested size cannot be zero");
                SB_THROW_IF(n_bytes > SIZE_MAX - (K_STORAGE_ALIGNMENT - 1),
                            "Requested size {} overflows when rounded up to {} bytes",
                            n_bytes, K_STORAGE_ALIGNMENT);

                SB_THROW_IF(device_index >= controls_.device_count(),
                            "Device index {} is out of range; the runtime drives {} devices",
                            device_index, controls_.device_count());

                const std::size_t size_bytes = (n_bytes + K_STORAGE_ALIGNMENT - 1) /
                                               K_STORAGE_ALIGNMENT * K_STORAGE_ALIGNMENT;

                SushiRuntime::Result<SushiRuntime::Buffer<std::byte>> allocated =
                    runtime.buffer<std::byte>(size_bytes, SushiRuntime::DeviceIndex{device_index},
                                              residency);
                SB_THROW_IF(!allocated, "Allocation failed for {} bytes: {}", size_bytes,
                            allocated.error().message);

                SB_THROW_IF(reinterpret_cast<std::uintptr_t>(allocated->data()) %
                                    K_STORAGE_ALIGNMENT != 0,
                            "The device returned an allocation that is not {}-byte aligned",
                            K_STORAGE_ALIGNMENT);

                buffer_ = std::move(*allocated);
            }

            Storage(const Storage&) = delete;
            Storage& operator=(const Storage&) = delete;

            /**
             * @brief Returns the base address of the whole allocation, ignoring every view's offset.
             * @return An address that is a multiple of K_STORAGE_ALIGNMENT.
             * @warning Never a tensor's data; use Tensor::data() to address elements.
             * @see modules/README.md
             */
            void* allocation_base() const noexcept { return buffer_.data(); }

            /**
             * @brief Returns the bytes allocated: the requested size rounded up to K_STORAGE_ALIGNMENT.
             */
            [[nodiscard]] std::size_t size_bytes() const noexcept { return buffer_.size(); }

            /** @brief Returns the queue of the device that owns the allocation. */
            [[nodiscard]] sycl::queue& queue() const
            {
                return controls_.queue(buffer_.device_index());
            }

            /** @brief Returns the kind of USM the allocation lives in (Shared, Device, or Host). */
            [[nodiscard]] SushiRuntime::Residency residency() const noexcept
            {
                return buffer_.residency();
            }

            /** @brief Returns the device queue that owns the allocation. */
            [[nodiscard]] std::size_t device_index() const noexcept
            {
                return buffer_.device_index().value();
            }

        protected:
            /** @brief Frees the allocation with the buffer that owns it. */
            ~Storage() override = default;

        private:
            /** @brief The runtime's device view, which hands out the queue of the allocation. */
            SushiRuntime::AdvancedControls controls_;

            /** @brief The buffer the runtime allocated; it frees the memory when it is destroyed. */
            SushiRuntime::Buffer<std::byte> buffer_;
    };
} // namespace SushiBLAS
