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
 * @file test_scratch_pool.cpp
 * @brief Tests that a ScratchPool keeps returned blocks idle, reuses them and trims them.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_ScratchPool, StartsWithoutIdleBlocks)
        {
            const ScratchPool pool(TestEnvironment::runtime());

            EXPECT_EQ(pool.idle_block_count(), std::size_t{0});
            EXPECT_TRUE(pool.shelf());
        }

        TEST(Unit_ScratchPool, AReturnedBlockIsReusedForTheSameSize)
        {
            ScratchPool pool(TestEnvironment::runtime());
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

        TEST(Unit_ScratchPool, TrimReleasesTheIdleBlocks)
        {
            ScratchPool pool(TestEnvironment::runtime());
            const std::size_t live_before = TestEnvironment::live_allocation_count();
            {
                const ScratchBuffer buffer(pool, 512, 0);
            }
            ASSERT_EQ(pool.idle_block_count(), std::size_t{1});

            pool.trim();

            EXPECT_EQ(pool.idle_block_count(), std::size_t{0});
            EXPECT_EQ(TestEnvironment::live_allocation_count(), live_before);
        }
    } // namespace Test
} // namespace SushiBLAS
