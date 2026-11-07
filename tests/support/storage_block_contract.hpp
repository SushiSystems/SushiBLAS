/**************************************************************************/
/* storage_block_contract.hpp                                             */
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
 * @file storage_block_contract.hpp
 * @brief Declares the checks every StorageBlock adapter passes, run by one typed suite per adapter.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/storage_block.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The byte the contract writes over a block.
        inline constexpr unsigned char K_STORAGE_BLOCK_CONTRACT_PATTERN = 0x5A;

        /** @brief Holds the checks that define a StorageBlock; an adapter's suite calls each one. */
        struct StorageBlockContract
        {
            /** @brief The size is the request rounded up to K_STORAGE_ALIGNMENT. */
            template<typename Subject>
            static void expect_size_rounds_up_to_the_alignment(const Subject& subject)
            {
                EXPECT_EQ(subject.block_of(1)->size_bytes(), K_STORAGE_ALIGNMENT);
                EXPECT_EQ(subject.block_of(K_STORAGE_ALIGNMENT)->size_bytes(), K_STORAGE_ALIGNMENT);
                EXPECT_EQ(subject.block_of(K_STORAGE_ALIGNMENT + 1)->size_bytes(),
                          2 * K_STORAGE_ALIGNMENT);
            }

            /** @brief The first byte is a multiple of K_STORAGE_ALIGNMENT. */
            template<typename Subject>
            static void expect_base_is_aligned(const Subject& subject)
            {
                const auto block = subject.block_of(1000);

                EXPECT_EQ(reinterpret_cast<std::uintptr_t>(block->data()) % K_STORAGE_ALIGNMENT,
                          std::uintptr_t{0});
            }

            /** @brief Every byte of the block can be written and read back in place. */
            template<typename Subject>
            static void expect_block_holds_what_is_written(const Subject& subject)
            {
                const auto block = subject.block_of(1000);
                const std::size_t size = block->size_bytes();

                std::memset(block->data(), K_STORAGE_BLOCK_CONTRACT_PATTERN, size);

                const auto* bytes = static_cast<const unsigned char*>(block->data());
                EXPECT_EQ(std::vector<unsigned char>(bytes, bytes + size),
                          std::vector<unsigned char>(size, K_STORAGE_BLOCK_CONTRACT_PATTERN));
            }

            /** @brief Bytes copied in at an offset are the bytes copied out from it. */
            template<typename Subject>
            static void expect_copies_round_trip_at_an_offset(const Subject& subject)
            {
                const auto block = subject.block_of(256);
                const std::vector<unsigned char> written{1, 2, 3, 4, 5, 6, 7, 8};
                std::vector<unsigned char> read(written.size());

                block->copy_from_host(written.data(), 16, written.size());
                block->copy_to_host(read.data(), 16, read.size());

                EXPECT_EQ(read, written);
            }

            /** @brief A copy lands in the bytes data() addresses. */
            template<typename Subject>
            static void expect_copies_reach_the_bytes_data_addresses(const Subject& subject)
            {
                const auto block = subject.block_of(256);
                const std::vector<unsigned char> written(32, K_STORAGE_BLOCK_CONTRACT_PATTERN);

                block->copy_from_host(written.data(), 64, written.size());

                const auto* bytes = static_cast<const unsigned char*>(block->data()) + 64;
                EXPECT_EQ(std::vector<unsigned char>(bytes, bytes + written.size()), written);
            }

            /** @brief A copy that passes the end of the block is refused, in both directions. */
            template<typename Subject>
            static void expect_copy_past_the_end_is_refused(const Subject& subject)
            {
                const auto block = subject.block_of(K_STORAGE_ALIGNMENT);
                std::vector<unsigned char> bytes(K_STORAGE_ALIGNMENT + 1);

                EXPECT_THROW(block->copy_to_host(bytes.data(), 0, bytes.size()),
                             std::runtime_error);
                EXPECT_THROW(block->copy_from_host(bytes.data(), 1, K_STORAGE_ALIGNMENT),
                             std::runtime_error);
                EXPECT_THROW(block->copy_to_host(bytes.data(), K_STORAGE_ALIGNMENT + 1, 0),
                             std::runtime_error);
            }

            /** @brief The block reports the residency and device its adapter places it in. */
            template<typename Subject>
            static void expect_reports_its_residency_and_device(const Subject& subject)
            {
                const auto block = subject.block_of(64);

                EXPECT_EQ(block->residency(), Subject::residency());
                EXPECT_EQ(block->device_index(), std::size_t{0});
            }

            /** @brief The block has a queue exactly when a device owns its memory. */
            template<typename Subject>
            static void expect_queue_follows_device_ownership(const Subject& subject)
            {
                const auto block = subject.block_of(64);

                EXPECT_EQ(block->queue() != nullptr, Subject::has_queue());
            }

            /** @brief A block that owns its memory names no owner. */
            template<typename Subject>
            static void expect_an_owned_block_names_no_owner(const Subject& subject)
            {
                const auto block = subject.block_of(64);

                EXPECT_FALSE(names_an_owner(block->owner_liveness()));
            }

            /** @brief A request of zero bytes is refused. */
            template<typename Subject>
            static void expect_empty_request_is_refused(const Subject& subject)
            {
                EXPECT_THROW((void)subject.block_of(0), std::runtime_error);
            }

            /** @brief A request whose rounded size overflows is refused. */
            template<typename Subject>
            static void expect_overflowing_request_is_refused(const Subject& subject)
            {
                EXPECT_THROW((void)subject.block_of(SIZE_MAX), std::runtime_error);
            }

            /** @brief Two live blocks share no byte. */
            template<typename Subject>
            static void expect_live_blocks_do_not_overlap(const Subject& subject)
            {
                const auto first = subject.block_of(256);
                const auto second = subject.block_of(256);

                const auto low = std::min(reinterpret_cast<std::uintptr_t>(first->data()),
                                          reinterpret_cast<std::uintptr_t>(second->data()));
                const auto high = std::max(reinterpret_cast<std::uintptr_t>(first->data()),
                                           reinterpret_cast<std::uintptr_t>(second->data()));
                EXPECT_GE(high - low, first->size_bytes());
            }
        };

    } // namespace Test
} // namespace SushiBLAS
