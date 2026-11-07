/**************************************************************************/
/* shared_allocator_contract.hpp                                          */
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
 * @file shared_allocator_contract.hpp
 * @brief Declares the checks every SharedAllocator adapter passes, run by one typed suite per adapter.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/host_device/shared_allocator.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The block size the contract requests, in bytes.
        inline constexpr std::size_t K_CONTRACT_BYTES = 1000;

        /// @brief The alignment the contract requests, in bytes; the one SharedArray asks for.
        inline constexpr std::size_t K_CONTRACT_ALIGNMENT = 128;

        /// @brief The byte the contract writes over a block.
        inline constexpr unsigned char K_CONTRACT_PATTERN = 0xA5;

        /** @brief Holds the checks that define a SharedAllocator; an adapter's suite calls each one. */
        struct SharedAllocatorContract
        {
            /** @brief A block starts at a multiple of the requested alignment. */
            static void expect_block_is_aligned(SharedAllocator& allocator)
            {
                void* block = allocator.allocate(K_CONTRACT_BYTES, K_CONTRACT_ALIGNMENT);

                ASSERT_NE(block, nullptr);
                EXPECT_EQ(reinterpret_cast<std::uintptr_t>(block) % K_CONTRACT_ALIGNMENT, std::uintptr_t{0});
                allocator.deallocate(block);
            }

            /** @brief The whole requested size can be written and read back. */
            static void expect_block_holds_what_is_written(SharedAllocator& allocator)
            {
                void* block = allocator.allocate(K_CONTRACT_BYTES, K_CONTRACT_ALIGNMENT);
                ASSERT_NE(block, nullptr);

                std::memset(block, K_CONTRACT_PATTERN, K_CONTRACT_BYTES);

                const auto* bytes = static_cast<const unsigned char*>(block);
                EXPECT_EQ(std::vector<unsigned char>(bytes, bytes + K_CONTRACT_BYTES),
                          std::vector<unsigned char>(K_CONTRACT_BYTES, K_CONTRACT_PATTERN));
                allocator.deallocate(block);
            }

            /** @brief Two blocks held at once do not share a byte. */
            static void expect_live_blocks_do_not_overlap(SharedAllocator& allocator)
            {
                void* first = allocator.allocate(K_CONTRACT_BYTES, K_CONTRACT_ALIGNMENT);
                void* second = allocator.allocate(K_CONTRACT_BYTES, K_CONTRACT_ALIGNMENT);
                ASSERT_NE(first, nullptr);
                ASSERT_NE(second, nullptr);

                const auto low = std::min(reinterpret_cast<std::uintptr_t>(first),
                                          reinterpret_cast<std::uintptr_t>(second));
                const auto high = std::max(reinterpret_cast<std::uintptr_t>(first),
                                           reinterpret_cast<std::uintptr_t>(second));
                EXPECT_GE(high - low, K_CONTRACT_BYTES);
                allocator.deallocate(first);
                allocator.deallocate(second);
            }

            /** @brief A block taken back leaves the allocator able to serve the same request again. */
            static void expect_a_request_is_served_again_after_a_block_is_taken_back(SharedAllocator& allocator)
            {
                allocator.deallocate(allocator.allocate(K_CONTRACT_BYTES, K_CONTRACT_ALIGNMENT));

                void* block = allocator.allocate(K_CONTRACT_BYTES, K_CONTRACT_ALIGNMENT);

                EXPECT_NE(block, nullptr);
                allocator.deallocate(block);
            }
        };
    } // namespace Test
} // namespace SushiBLAS
