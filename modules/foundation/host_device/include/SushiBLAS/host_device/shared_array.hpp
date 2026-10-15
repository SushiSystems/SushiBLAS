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
 * @file shared_array.hpp
 * @brief Declares SharedArray, one aligned shared USM array freed when it goes out of scope.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <new>
#include <vector>
#include <sycl/sycl.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds the alignment, in bytes, of every array a SharedArray allocates. */
        inline constexpr std::size_t K_SHARED_ARRAY_ALIGNMENT = 128;

        /**
         * @brief Owns one aligned shared USM array that host code and a kernel both read.
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
                    : queue_(queue), count_(count),
                      data_(sycl::aligned_alloc_shared<T>(K_SHARED_ARRAY_ALIGNMENT,
                                                          count ? count : 1, queue))
                {
                    if (data_ == nullptr)
                        throw std::bad_alloc();
                    std::fill(data_, data_ + count_, T{});
                }

                /** @brief Frees the array. */
                ~SharedArray()
                {
                    sycl::free(data_, queue_);
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
                /** @brief The queue whose context owns the array. */
                sycl::queue& queue_;

                /** @brief The element count. */
                std::size_t count_;

                /** @brief The first element. */
                T* data_;
        };
    } // namespace Test
} // namespace SushiBLAS
