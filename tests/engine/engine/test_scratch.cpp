/**************************************************************************/
/* test_scratch.cpp                                                       */
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
 * @file test_scratch.cpp
 * @brief Tests the lease a ScratchBuffer holds: its bytes, its region, its copies and its return.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <utility>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Medium_Scratch, AnEmptyBufferHoldsNothing)
        {
            const ScratchBuffer empty;

            EXPECT_TRUE(empty.is_empty());
            EXPECT_EQ(empty.data(), nullptr);
            EXPECT_EQ(empty.bytes(), std::size_t{0});
            EXPECT_FALSE(empty.storage());
        }

        TEST(Medium_Scratch, ALeaseNamesItsBytesAndItsRegion)
        {
            ScratchPool pool(TestEnvironment::runtime());
            const ScratchBuffer buffer(pool, 1000, 0);

            EXPECT_FALSE(buffer.is_empty());
            EXPECT_NE(buffer.data(), nullptr);
            EXPECT_EQ(buffer.bytes(), std::size_t{1000});
            EXPECT_TRUE(buffer.storage());
            EXPECT_GE(buffer.storage()->size_bytes(), std::size_t{1000});
        }

        TEST(Medium_Scratch, TheBlockReturnsToTheShelfWhenTheLastCopyGoes)
        {
            ScratchPool pool(TestEnvironment::runtime());
            const std::size_t idle_before = pool.idle_block_count();
            {
                const ScratchBuffer first(pool, 256, 0);
                ScratchBuffer copy = first;
                ScratchBuffer moved = std::move(copy);
                EXPECT_EQ(moved.data(), first.data());
                EXPECT_TRUE(copy.is_empty());
                EXPECT_EQ(pool.idle_block_count(), idle_before);
            }

            EXPECT_EQ(pool.idle_block_count(), idle_before + 1);
        }
    } // namespace Test
} // namespace SushiBLAS
