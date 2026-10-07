/**************************************************************************/
/* test_scratch_shelf.cpp                                                 */
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
 * @file test_scratch_shelf.cpp
 * @brief Tests the shelf of idle device scratch blocks and its size classes.
 * @author Mustafa Garip
 */

#include <array>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/scratch_shelf.hpp>
#include <SushiRuntime/SushiRuntime.h>
#include "../../common/test_environment.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Builds one empty shelf over the test environment's context for each test. */
        class Unit_ScratchShelf : public ::testing::Test
        {
            protected:
                /** @brief Returns the context the shelf allocates from. */
                [[nodiscard]] static SushiRuntime::Execution::RuntimeContext& context()
                {
                    return TestEnvironment::get_context();
                }

                /// @brief The shelf under test; empty when a test starts.
                SushiRuntime::sushi_ptr<Detail::ScratchShelf> shelf_ =
                    SushiRuntime::make_sushi<Detail::ScratchShelf>(context());
        };

        TEST_F(Unit_ScratchShelf, ARequestIsRoundedUpToItsClass)
        {
            const std::array<std::size_t, 4> requests = {1, 128, 129, 4096};
            const std::array<std::size_t, 4> expected = {128, 128, 256, 4096};

            for (std::size_t i = 0; i < requests.size(); ++i)
            {
                Detail::ScratchBlock* block = shelf_->take(requests[i], 0);
                ASSERT_NE(block, nullptr);
                EXPECT_EQ(Detail::scratch_class_bytes(block->size_class), expected[i])
                    << "request of " << requests[i] << " bytes";
                shelf_->give_back(block);
            }
        }

        TEST_F(Unit_ScratchShelf, AReturnedBlockIsLentAgain)
        {
            Detail::ScratchBlock* first = shelf_->take(128, 0);
            const void* first_base = first->storage->allocation_base();
            shelf_->give_back(first);
            const std::size_t live_after_return = context().live_usm_count();

            Detail::ScratchBlock* second = shelf_->take(128, 0);
            EXPECT_EQ(second->storage->allocation_base(), first_base);
            EXPECT_EQ(context().live_usm_count(), live_after_return);
            shelf_->give_back(second);
        }

        TEST_F(Unit_ScratchShelf, ARequestAboveTheLargestClassIsNotPooled)
        {
            const std::size_t live_baseline = context().live_usm_count();
            const std::size_t idle_baseline = shelf_->idle_block_count();

            Detail::ScratchBlock* block = shelf_->take(4097, 0);
            EXPECT_EQ(block->size_class, Detail::K_SCRATCH_UNPOOLED);
            shelf_->give_back(block);

            EXPECT_EQ(shelf_->idle_block_count(), idle_baseline);
            EXPECT_EQ(context().live_usm_count(), live_baseline);
        }

        TEST_F(Unit_ScratchShelf, AFullClassFreesWhatItCannotKeep)
        {
            const std::size_t live_baseline = context().live_usm_count();
            const std::size_t taken = Detail::K_SCRATCH_IDLE_BLOCKS_PER_CLASS + 1;

            std::vector<Detail::ScratchBlock*> blocks;
            blocks.reserve(taken);
            for (std::size_t i = 0; i < taken; ++i)
                blocks.push_back(shelf_->take(512, 0));
            for (Detail::ScratchBlock* block : blocks)
                shelf_->give_back(block);

            EXPECT_EQ(shelf_->idle_block_count(), std::size_t{64});
            EXPECT_EQ(context().live_usm_count(), live_baseline + 64);
        }

        TEST_F(Unit_ScratchShelf, TrimFreesEveryIdleBlock)
        {
            const std::size_t live_baseline = context().live_usm_count();

            Detail::ScratchBlock* narrow = shelf_->take(128, 0);
            Detail::ScratchBlock* wide = shelf_->take(4096, 0);
            shelf_->give_back(narrow);
            shelf_->give_back(wide);
            ASSERT_EQ(shelf_->idle_block_count(), std::size_t{2});

            shelf_->trim();

            EXPECT_EQ(shelf_->idle_block_count(), std::size_t{0});
            EXPECT_EQ(context().live_usm_count(), live_baseline);
        }

        TEST_F(Unit_ScratchShelf, AnUnknownDeviceIsRefused)
        {
            const std::size_t live_baseline = context().live_usm_count();

            EXPECT_THROW((void)shelf_->take(128, context().device_count()), std::runtime_error);
            EXPECT_EQ(context().live_usm_count(), live_baseline);
        }
    } // namespace Test
} // namespace SushiBLAS
