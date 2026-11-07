/**************************************************************************/
/* host_block.hpp                                                         */
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
 * @file host_block.hpp
 * @brief Declares HostBlock, aligned host memory that no SushiRuntime::Runtime owns.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <new>

namespace SushiBLAS
{
    /**
     * @class HostBlock
     * @brief Owns one aligned block of ordinary host memory and frees it with itself.
     * @see modules/data/tensor/README.md
     */
    class HostBlock final
    {
        public:
            /** @brief Constructs a block that owns no memory. */
            HostBlock() noexcept = default;

            /**
             * @brief Allocates n_bytes of host memory at an address aligned to alignment.
             * @param n_bytes Number of bytes; above zero.
             * @param alignment A power of two.
             * @throws std::bad_alloc if the allocation fails.
             */
            HostBlock(std::size_t n_bytes, std::size_t alignment)
                : data_(::operator new(n_bytes, std::align_val_t{alignment})),
                  size_bytes_(n_bytes),
                  alignment_(alignment)
            {

            }

            HostBlock(const HostBlock&) = delete;
            HostBlock& operator=(const HostBlock&) = delete;

            /** @brief Takes the memory of other, which owns none afterwards. */
            HostBlock(HostBlock&& other) noexcept
                : data_(other.data_), size_bytes_(other.size_bytes_), alignment_(other.alignment_)
            {
                other.data_ = nullptr;
                other.size_bytes_ = 0;
            }

            /** @brief Frees the memory it owns, then takes the memory of other. */
            HostBlock& operator=(HostBlock&& other) noexcept
            {
                if (this != &other)
                {
                    release();
                    data_ = other.data_;
                    size_bytes_ = other.size_bytes_;
                    alignment_ = other.alignment_;
                    other.data_ = nullptr;
                    other.size_bytes_ = 0;
                }
                return *this;
            }

            /** @brief Frees the memory it owns. */
            ~HostBlock()
            {
                release();
            }

            /** @brief Returns the first byte, or nullptr for a block that owns no memory. */
            [[nodiscard]] void* data() const noexcept { return data_; }

            /** @brief Returns the bytes owned; 0 for a block that owns no memory. */
            [[nodiscard]] std::size_t size_bytes() const noexcept { return size_bytes_; }

        private:
            /** @brief Frees the memory and leaves the block empty. */
            void release() noexcept
            {
                if (data_ != nullptr)
                    ::operator delete(data_, std::align_val_t{alignment_});
                data_ = nullptr;
            }

            /// @brief The first byte of the block; null when it owns none.
            void* data_ = nullptr;

            /// @brief The bytes owned.
            std::size_t size_bytes_ = 0;

            /// @brief The alignment the block was allocated with, which its release needs.
            std::size_t alignment_ = alignof(std::max_align_t);
    };
} // namespace SushiBLAS
