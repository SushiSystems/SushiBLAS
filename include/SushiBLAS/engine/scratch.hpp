/**************************************************************************/
/* scratch.hpp                                                            */
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
 * @file scratch.hpp
 * @brief Declares ScratchBuffer, the device scratch allocation one operation owns.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/storage.hpp>
#include <SushiRuntime/core/resource_region.hpp>

namespace SushiBLAS
{
    /**
     * @class ScratchBuffer
     * @brief Owns a device-resident scratch allocation, shared by every copy of the handle.
     * @see include/SushiBLAS/engine/README.md
     */
    class ScratchBuffer final
    {
        public:
            /** @brief Builds an empty handle that owns nothing. */
            ScratchBuffer() noexcept = default;

            /**
             * @brief Allocates @p bytes of device memory, or nothing when @p bytes is zero.
             * @param engine Engine whose runtime context allocates; the context must outlive
             *        every copy.
             * @param device_index Device queue that owns the allocation: the operand's
             *        Storage::get_device_index().
             * @throws std::runtime_error if the allocation fails.
             */
            ScratchBuffer(Engine& engine, std::size_t bytes, std::size_t device_index)
                : bytes_(bytes)
            {
                if (bytes_ == 0)
                    return;

                storage_ = SushiRuntime::make_sushi<Storage>(engine.get_context(), bytes_,
                                                             SushiRuntime::API::Residency::Device,
                                                             device_index);
            }

            /** @brief Returns the base address, or null when the handle owns nothing. */
            [[nodiscard]] void* data() const noexcept
            {
                return storage_ ? storage_->allocation_base() : nullptr;
            }

            /** @brief Returns the usable size in bytes; 0 when the handle owns nothing. */
            [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

            /** @brief Returns true when the handle owns no allocation. */
            [[nodiscard]] bool is_empty() const noexcept { return bytes_ == 0; }

            /**
             * @brief Returns the region a task that writes the scratch declares.
             * @pre is_empty() is false.
             */
            [[nodiscard]] SushiRuntime::Core::ResourceRegion region() const noexcept
            {
                return {data(), 0, bytes_};
            }

        private:
            /** @brief The shared allocation, or null when bytes_ is zero. */
            SushiRuntime::sushi_ptr<Storage> storage_;

            /** @brief The size the caller asked for, in bytes. */
            std::size_t bytes_ = 0;
    };
} // namespace SushiBLAS
