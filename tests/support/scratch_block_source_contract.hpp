/**************************************************************************/
/* scratch_block_source_contract.hpp                                      */
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
 * @file scratch_block_source_contract.hpp
 * @brief Declares the checks every ScratchBlockSource adapter passes, run by one typed suite per adapter.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/scratch_block_source.hpp>
#include <SushiBLAS/tensor/storage.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Serves count blocks of bytes each and keeps them alive; returns the keepers. */
        [[nodiscard]] inline std::vector<SushiRuntime::sushi_ptr<Storage>> serve_live_blocks(
            ScratchBlockSource& source, std::size_t count, std::size_t bytes)
        {
            std::vector<SushiRuntime::sushi_ptr<Storage>> blocks;
            blocks.reserve(count);
            for (std::size_t i = 0; i < count; ++i)
                blocks.push_back(source.serve(bytes, 0));
            return blocks;
        }

        /** @brief Returns the base addresses of blocks. */
        [[nodiscard]] inline std::unordered_set<const void*> base_addresses_of(
            const std::vector<SushiRuntime::sushi_ptr<Storage>>& blocks)
        {
            std::unordered_set<const void*> addresses;
            for (const auto& block : blocks)
                addresses.insert(block->allocation_base());
            return addresses;
        }

        /** @brief Returns whether serving throws std::runtime_error. */
        [[nodiscard]] inline bool serving_is_refused(ScratchBlockSource& source, std::size_t bytes,
                                                     std::size_t device_index)
        {
            try
            {
                (void)source.serve(bytes, device_index);
            }
            catch (const std::runtime_error&)
            {
                return true;
            }
            return false;
        }

        /** @brief Holds the checks that define a ScratchBlockSource; an adapter's suite calls each one. */
        struct BlockSourceContract
        {
            /** @brief Serves bytes and expects a block of at least that size. */
            template<typename Subject>
            static void expect_serves_at_least(Subject& subject, std::size_t bytes)
            {
                const auto block = subject.port().serve(bytes, 0);

                ASSERT_NE(block.get(), nullptr);
                EXPECT_GE(block->size_bytes(), bytes);
            }

            /** @brief A request of one byte, of one byte over an alignment step and of several steps is served whole. */
            template<typename Subject>
            static void expect_block_holds_the_request(Subject& subject)
            {
                expect_serves_at_least(subject, 1);
                expect_serves_at_least(subject, K_STORAGE_ALIGNMENT + 1);
                expect_serves_at_least(subject, 5 * K_STORAGE_ALIGNMENT);
            }

            /** @brief The base of a served block is a multiple of K_STORAGE_ALIGNMENT. */
            template<typename Subject>
            static void expect_base_is_aligned(Subject& subject)
            {
                const auto block = subject.port().serve(1000, 0);

                EXPECT_EQ(reinterpret_cast<std::uintptr_t>(block->allocation_base()) %
                              K_STORAGE_ALIGNMENT,
                          std::uintptr_t{0});
            }

            /** @brief Blocks that are alive together have different base addresses. */
            template<typename Subject>
            static void expect_live_blocks_do_not_share_an_address(Subject& subject)
            {
                constexpr std::size_t K_COUNT = 40;
                const auto blocks = serve_live_blocks(subject.port(), K_COUNT, 128);

                EXPECT_EQ(base_addresses_of(blocks).size(), K_COUNT);
            }

            /** @brief The source serves at least one device. */
            template<typename Subject>
            static void expect_serves_at_least_one_device(Subject& subject)
            {
                EXPECT_GE(subject.port().device_count(), std::size_t{1});
            }

            /** @brief A device index at device_count() is refused. */
            template<typename Subject>
            static void expect_unknown_device_is_refused(Subject& subject)
            {
                EXPECT_TRUE(serving_is_refused(subject.port(), 128, subject.port().device_count()));
            }

            /** @brief A request of zero bytes is refused. */
            template<typename Subject>
            static void expect_zero_bytes_are_refused(Subject& subject)
            {
                EXPECT_TRUE(serving_is_refused(subject.port(), 0, 0));
            }
        };
    } // namespace Test
} // namespace SushiBLAS
