/**************************************************************************/
/* shared_array.hpp                                                       */
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
 * @file shared_array.hpp
 * @brief Declares SharedArray, one aligned shared USM array freed when it goes out of scope.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <new>
#include <optional>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/host_device/shared_allocator.hpp>
#include <SushiBLAS/host_device/usm_shared_allocator.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds the alignment, in bytes, of every array a SharedArray allocates. */
        inline constexpr std::size_t K_SHARED_ARRAY_ALIGNMENT = 128;

        /**
         * @brief Owns one aligned array that host code reads and, with the default allocator, a kernel too.
         *
         * The queue constructors take shared USM from a UsmSharedAllocator on that queue. The
         * allocator constructors take the memory from any SharedAllocator, which must outlive the array.
         * @tparam T The element type; trivially copyable.
         */
        template<typename T>
        class SharedArray final
        {
            public:
                /** @brief Allocates one element per value on queue and copies the values in. */
                SharedArray(sycl::queue& queue, const std::vector<T>& values)
                    : SharedArray(queue, values.size())
                {
                    std::copy(values.begin(), values.end(), data_);
                }

                /** @brief Allocates count value-initialised elements on the device of queue. */
                SharedArray(sycl::queue& queue, std::size_t count)
                    : owned_(std::in_place, queue), allocator_(*owned_), count_(count),
                      data_(allocate())
                {

                }

                /** @brief Allocates one element per value from allocator and copies the values in. */
                SharedArray(SharedAllocator& allocator, const std::vector<T>& values)
                    : SharedArray(allocator, values.size())
                {
                    std::copy(values.begin(), values.end(), data_);
                }

                /** @brief Allocates count value-initialised elements from allocator. */
                SharedArray(SharedAllocator& allocator, std::size_t count)
                    : allocator_(allocator), count_(count), data_(allocate())
                {

                }

                /** @brief Frees the array. */
                ~SharedArray()
                {
                    allocator_.deallocate(data_);
                }

                SharedArray(const SharedArray&) = delete;
                SharedArray& operator=(const SharedArray&) = delete;
                SharedArray(SharedArray&&) = delete;
                SharedArray& operator=(SharedArray&&) = delete;

                /** @brief Returns the first element. */
                [[nodiscard]] T* get() const noexcept
                {
                    return data_;
                }

                /** @brief Returns the element count. */
                [[nodiscard]] std::size_t size() const noexcept
                {
                    return count_;
                }

                /** @brief Returns a host copy of the elements. */
                [[nodiscard]] std::vector<T> values() const
                {
                    return std::vector<T>(data_, data_ + count_);
                }

            private:
                /** @brief Takes the block for count_ elements, at least one, and value-initialises them. */
                [[nodiscard]] T* allocate() const
                {
                    T* block = static_cast<T*>(allocator_.allocate(
                        (count_ ? count_ : 1) * sizeof(T), K_SHARED_ARRAY_ALIGNMENT));
                    if (block == nullptr)
                        throw std::bad_alloc();
                    std::fill(block, block + count_, T{});
                    return block;
                }

                /** @brief The allocator of the queue constructors; empty for the others. */
                std::optional<UsmSharedAllocator> owned_;

                /** @brief The allocator the block came from and goes back to. */
                SharedAllocator& allocator_;

                /** @brief The element count. */
                std::size_t count_;

                /** @brief The first element. */
                T* data_;
        };
    } // namespace Test
} // namespace SushiBLAS
