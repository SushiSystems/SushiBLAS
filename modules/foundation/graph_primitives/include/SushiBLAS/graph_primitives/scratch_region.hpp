/**************************************************************************/
/* scratch_region.hpp                                                     */
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
 * @file scratch_region.hpp
 * @brief Declares ScratchRegion, the handle-named windows of one scratch block a primitive keeps.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <utility>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/core/error.hpp>
#include <SushiRuntime/core/result.hpp>
#include <SushiRuntime/region/buffer_region.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        namespace Detail
        {
            /**
             * @brief Names windows of one scratch block of T by the handle the recorder gave it.
             *
             * name_scratch() asks the recorder for the handle once; every window is cut from it.
             * @tparam T The element type of the block.
             * @see modules/foundation/graph_primitives/README.md
             */
            template <typename T>
            class ScratchRegion
            {
                public:
                    /** @brief Names no block; every window is an empty region. */
                    ScratchRegion() noexcept = default;

                    /** @brief Names the block whole holds. */
                    explicit ScratchRegion(const SushiRuntime::BufferRegion& whole) noexcept
                        : whole_(whole)
                    {
                    }

                    /** @brief Returns the whole block as a region. */
                    [[nodiscard]] const SushiRuntime::BufferRegion& whole() const noexcept
                    {
                        return whole_;
                    }

                    /**
                     * @brief Returns elements [offset, offset + count) of the block as a region.
                     * @return An empty region when the block is not named.
                     */
                    [[nodiscard]] SushiRuntime::BufferRegion window(std::size_t offset,
                                                                    std::size_t count) const noexcept
                    {
                        return SushiRuntime::BufferRegion(whole_.base, offset * sizeof(T),
                                                          count * sizeof(T));
                    }

                private:
                    /** @brief The whole block; empty when none is named. */
                    SushiRuntime::BufferRegion whole_;
            };

            /**
             * @brief Asks the recorder for the handle of the scratch block at base.
             * @param recorder The port that handed the block out; it names the block through borrowed_region().
             * @param base The block's first element; null names no block.
             * @param count The elements the block holds.
             * @param device The device the block was allocated on.
             * @return The windows of the block, none when base is null or count is zero, or the recorder's error.
             */
            template <class Port, typename T>
            [[nodiscard]] SushiRuntime::Result<ScratchRegion<T>>
            name_scratch(Port& recorder, const T* base, std::size_t count,
                         SushiRuntime::DeviceIndex device)
            {
                if (base == nullptr || count == 0)
                    return ScratchRegion<T>();

                SushiRuntime::Result<SushiRuntime::BufferRegion> named =
                    recorder.borrowed_region(base, count * sizeof(T), device);
                if (!named)
                    return SushiRuntime::make_error(std::move(named.error()));
                return ScratchRegion<T>(*named);
            }
        } // namespace Detail
    } // namespace GraphPrimitives
} // namespace SushiBLAS
