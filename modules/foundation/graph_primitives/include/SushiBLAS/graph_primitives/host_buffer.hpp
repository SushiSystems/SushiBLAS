/**************************************************************************/
/* host_buffer.hpp                                                        */
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

/**
 * @file host_buffer.hpp
 * @brief Declares HostBuffer, the operand type of HostPrimitiveRecorder.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/handle/global_handle.hpp>
#include <SushiRuntime/handle/local_handle_space.hpp>
#include <SushiRuntime/region/buffer_region.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        namespace Detail
        {
            /// @brief The space id of the handles of the host double; the tensor module's host
            ///        blocks use 1, so a handle of one never equals a handle of the other.
            inline constexpr std::uint16_t K_HOST_DOUBLE_SPACE_ID = 2;

            /**
             * @brief Returns the space that mints the handle of every HostBuffer and every scratch
             *        block of HostPrimitiveRecorder in the process.
             *
             * These blocks are memory no runtime holds, so no directory reads their handles and no
             * runtime graph is given one.
             * @see modules/foundation/graph_primitives/README.md
             */
            [[nodiscard]] inline SushiRuntime::LocalHandleSpace& host_double_space() noexcept
            {
                static SushiRuntime::LocalHandleSpace space(K_HOST_DOUBLE_SPACE_ID);
                return space;
            }

            /**
             * @brief Returns a new handle from host_double_space().
             * @throws std::runtime_error if the space has minted 2^32 handles.
             */
            [[nodiscard]] inline SushiRuntime::GlobalHandle mint_double_handle()
            {
                const SushiRuntime::MintOutcome outcome =
                    host_double_space().mint(SushiRuntime::HandleKind::Buffer);
                if (outcome.status != SushiRuntime::HandleStatus::Ok)
                    throw std::runtime_error("host double: the handle space is exhausted");
                return outcome.handle;
            }
        } // namespace Detail

        /**
         * @brief Owns count zeroed elements of host memory and offers the surface a primitive reads.
         *
         * A primitive reads size(), data(), device_index(), residency() and region() of its operands;
         * this type gives them without a SushiRuntime::Runtime.
         * @see modules/foundation/graph_primitives/README.md
         */
        template <typename T>
        class HostBuffer
        {
            public:
                /** @brief Allocates count zeroed elements. */
                explicit HostBuffer(std::size_t count)
                    : data_(new T[count == 0 ? 1 : count]()), size_(count)
                {
                }

                /** @brief Returns element i; the caller keeps i below size(). */
                [[nodiscard]] T& operator[](std::size_t i) const noexcept { return data_[i]; }

                /** @brief Returns the number of elements. */
                [[nodiscard]] std::size_t size() const noexcept { return size_; }

                /** @brief Returns the address of element 0. */
                [[nodiscard]] T* data() const noexcept { return data_.get(); }

                /** @brief Returns the one device a host buffer belongs to: the default index. */
                [[nodiscard]] std::optional<SushiRuntime::DeviceIndex> device_index() const noexcept
                {
                    return SushiRuntime::DeviceIndex{};
                }

                /** @brief Returns the residency a host buffer keeps its memory in. */
                [[nodiscard]] SushiRuntime::Residency residency() const noexcept
                {
                    return SushiRuntime::Residency::Host;
                }

                /** @brief Returns the whole buffer as a region, as a dependency set keys it. */
                [[nodiscard]] SushiRuntime::BufferRegion region() const noexcept
                {
                    return SushiRuntime::BufferRegion(handle_);
                }

            private:
                /** @brief The elements; the address stays when the buffer moves. */
                std::unique_ptr<T[]> data_;

                /** @brief The number of elements. */
                std::size_t size_ = 0;

                /** @brief The handle that names the buffer; it moves with the buffer. */
                SushiRuntime::GlobalHandle handle_ = Detail::mint_double_handle();
        };

    } // namespace GraphPrimitives
} // namespace SushiBLAS
