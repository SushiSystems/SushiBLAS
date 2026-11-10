/**************************************************************************/
/* test_scratch_pool.cpp                                                  */
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
 * @file test_scratch_pool.cpp
 * @brief Tests that a ScratchPool keeps returned blocks idle, reuses them and trims them.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/host_scratch_source.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_ScratchPool, AReturnedBlockIsReusedForTheSameSize)
        {
            HostScratchSource pool_source;
            ScratchPool pool(pool_source);
            void* first_address = nullptr;
            {
                const ScratchBuffer first(pool, 4096, 0);
                first_address = first.data();
            }
            ASSERT_EQ(pool.idle_block_count(), std::size_t{1});

            const ScratchBuffer second(pool, 4096, 0);

            EXPECT_EQ(second.data(), first_address);
            EXPECT_EQ(pool.idle_block_count(), std::size_t{0});
        }

        TEST(Small_ScratchPool, TrimReleasesTheIdleBlocks)
        {
            HostScratchSource pool_source;
            ScratchPool pool(pool_source);
            {
                const ScratchBuffer buffer(pool, 512, 0);
            }
            ASSERT_EQ(pool.idle_block_count(), std::size_t{1});
            ASSERT_EQ(pool_source.live_blocks(), std::size_t{1});

            pool.trim();

            EXPECT_EQ(pool.idle_block_count(), std::size_t{0});
            EXPECT_EQ(pool_source.live_blocks(), std::size_t{0});
        }
    } // namespace Test
} // namespace SushiBLAS
