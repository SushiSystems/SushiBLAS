/**************************************************************************/
/* test_host_scratch_source.cpp                                           */
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
 * @file test_host_scratch_source.cpp
 * @brief Tests what HostScratchSource counts and the memory it serves.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/host_scratch_source.hpp>
#include "support/scratch_block_source_contract.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Tests that a new host source has served nothing. */
        TEST(Small_HostScratchSource, ItStartsWithNothingServed)
        {
            const HostScratchSource source;

            EXPECT_EQ(source.served_blocks(), std::size_t{0});
            EXPECT_EQ(source.served_bytes(), std::size_t{0});
        }

        /** @brief Tests that the counters add each served block and its rounded size. */
        TEST(Small_HostScratchSource, TheCountersAddEveryServedBlock)
        {
            HostScratchSource source;

            const auto first = source.serve(1, 0);
            const auto second = source.serve(K_STORAGE_ALIGNMENT + 1, 0);

            EXPECT_EQ(source.served_blocks(), std::size_t{2});
            EXPECT_EQ(source.served_bytes(), first->size_bytes() + second->size_bytes());
            EXPECT_EQ(source.served_bytes(), 3 * K_STORAGE_ALIGNMENT);
        }

        /** @brief Tests that a refused request is not counted. */
        TEST(Small_HostScratchSource, ARefusedRequestIsNotCounted)
        {
            HostScratchSource source;

            EXPECT_TRUE(serving_is_refused(source, 128, 1));
            EXPECT_TRUE(serving_is_refused(source, 0, 0));

            EXPECT_EQ(source.served_blocks(), std::size_t{0});
            EXPECT_EQ(source.served_bytes(), std::size_t{0});
        }

        /** @brief Tests that a served block is host memory. */
        TEST(Small_HostScratchSource, AServedBlockIsHostMemory)
        {
            HostScratchSource source;

            const auto block = source.serve(256, 0);

            EXPECT_EQ(block->block().queue(), nullptr);
            EXPECT_EQ(block->residency(), SushiRuntime::Residency::Host);
        }
    } // namespace Test
} // namespace SushiBLAS
