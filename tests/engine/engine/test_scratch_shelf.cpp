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
 * @file test_scratch_shelf.cpp
 * @brief Tests the shelf of idle device scratch blocks and its size classes.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/host_scratch_source.hpp>
#include <SushiBLAS/engine/scratch_shelf.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Names a request and the size class it must be rounded up to. */
            struct ClassRow
            {
                const char* name;      /**< The row name a failure shows. */
                std::size_t request;   /**< The bytes asked for. */
                std::size_t expected;  /**< The bytes of the class that serves it. */
            };

            /** @brief Returns the requests of 1, 128, 129 and 4096 bytes. */
            std::vector<ClassRow> class_rows()
            {
                return {{"OneByte", 1, 128},
                        {"ExactlyTheSmallestClass", 128, 128},
                        {"OneByteOverTheSmallestClass", 129, 256},
                        {"TheLargestClass", 4096, 4096}};
            }

            /** @brief Names whether an idle block waits before the lease and what the counters read after. */
            struct HitRow
            {
                const char* name;     /**< The row name a failure shows. */
                bool idle_first;      /**< Whether a block of the class is returned before the lease. */
                std::uint64_t hits;   /**< The pool hits after the lease. */
            };

            /** @brief Returns the lease that finds its class empty and the one that finds an idle block. */
            std::vector<HitRow> hit_rows()
            {
                return {{"ALeaseThatFindsItsClassEmptyIsAMiss", false, 0},
                        {"ALeaseAnIdleBlockAnswersIsAHit", true, 1}};
            }
        } // namespace

        /** @brief Builds one empty shelf over a host source for each test. */
        class Small_ScratchShelf : public ::testing::Test
        {
            protected:
                /** @brief Takes count blocks of bytes bytes each from the shelf. */
                std::vector<SushiBLAS::Detail::ScratchBlock*> take_blocks(std::size_t count,
                                                                         std::size_t bytes)
                {
                    std::vector<SushiBLAS::Detail::ScratchBlock*> blocks;
                    blocks.reserve(count);
                    for (std::size_t index = 0; index < count; ++index)
                        blocks.push_back(shelf_->take(bytes, 0));
                    return blocks;
                }

                /** @brief Gives every block back to the shelf. */
                void give_back_all(const std::vector<SushiBLAS::Detail::ScratchBlock*>& blocks)
                {
                    for (SushiBLAS::Detail::ScratchBlock* block : blocks)
                        shelf_->give_back(block);
                }

                /// @brief The source that serves every block of the shelf.
                HostScratchSource source_;

                /// @brief The shelf under test; empty when a test starts.
                SushiRuntime::sushi_ptr<SushiBLAS::Detail::ScratchShelf> shelf_ =
                    SushiRuntime::make_sushi<SushiBLAS::Detail::ScratchShelf>(source_);
        };

        /** @brief Runs one row of class_rows() on an empty shelf. */
        class Small_ScratchShelfClass : public Small_ScratchShelf,
                                        public ::testing::WithParamInterface<ClassRow>
        {

        };

        TEST_P(Small_ScratchShelfClass, ARequestIsRoundedUpToItsClass)
        {
            SushiBLAS::Detail::ScratchBlock* block = shelf_->take(GetParam().request, 0);
            ASSERT_NE(block, nullptr);

            EXPECT_EQ(SushiBLAS::Detail::scratch_class_bytes(block->size_class),
                      GetParam().expected);
            shelf_->give_back(block);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_ScratchShelfClass, ::testing::ValuesIn(class_rows()),
            [](const ::testing::TestParamInfo<ClassRow>& info)
            { return std::string(info.param.name); });

        TEST_F(Small_ScratchShelf, AReturnedBlockIsLentAgain)
        {
            SushiBLAS::Detail::ScratchBlock* first = shelf_->take(128, 0);
            const void* first_base = first->storage->allocation_base();
            shelf_->give_back(first);
            const std::size_t served_after_return = source_.served_blocks();

            SushiBLAS::Detail::ScratchBlock* second = shelf_->take(128, 0);

            EXPECT_EQ(second->storage->allocation_base(), first_base);
            EXPECT_EQ(source_.served_blocks(), served_after_return);
            shelf_->give_back(second);
        }

        TEST_F(Small_ScratchShelf, ARequestAboveTheLargestClassIsNotPooled)
        {
            SushiBLAS::Detail::ScratchBlock* block = shelf_->take(4097, 0);
            EXPECT_EQ(block->size_class, SushiBLAS::Detail::K_SCRATCH_UNPOOLED);
            SushiRuntime::sushi_ptr<Storage> storage = block->storage;

            shelf_->give_back(block);

            EXPECT_EQ(shelf_->idle_block_count(), std::size_t{0});
            EXPECT_EQ(storage->ref_count(), 1);
        }

        TEST_F(Small_ScratchShelf, AFullClassFreesWhatItCannotKeep)
        {
            const std::size_t taken = SushiBLAS::Detail::K_SCRATCH_IDLE_BLOCKS_PER_CLASS + 1;

            give_back_all(take_blocks(taken, 512));

            EXPECT_EQ(shelf_->idle_block_count(),
                      SushiBLAS::Detail::K_SCRATCH_IDLE_BLOCKS_PER_CLASS);
            EXPECT_EQ(source_.served_blocks(), taken);
        }

        TEST_F(Small_ScratchShelf, TrimFreesEveryIdleBlock)
        {
            SushiBLAS::Detail::ScratchBlock* narrow = shelf_->take(128, 0);
            SushiBLAS::Detail::ScratchBlock* wide = shelf_->take(4096, 0);
            SushiRuntime::sushi_ptr<Storage> narrow_storage = narrow->storage;
            SushiRuntime::sushi_ptr<Storage> wide_storage = wide->storage;
            shelf_->give_back(narrow);
            shelf_->give_back(wide);
            ASSERT_EQ(shelf_->idle_block_count(), std::size_t{2});

            shelf_->trim();

            EXPECT_EQ(shelf_->idle_block_count(), std::size_t{0});
            EXPECT_EQ(narrow_storage->ref_count(), 1);
            EXPECT_EQ(wide_storage->ref_count(), 1);
        }

        TEST_F(Small_ScratchShelf, AnUnknownDeviceIsRefusedBeforeAnyAllocation)
        {
            EXPECT_THROW((void)shelf_->take(128, source_.device_count()), std::runtime_error);
            EXPECT_EQ(source_.served_blocks(), std::size_t{0});
        }

        TEST_F(Small_ScratchShelf, ALeaseAboveTheLargestClassIsCountedAsUnpooled)
        {
            const std::size_t above = SushiBLAS::Detail::K_SCRATCH_MAX_BLOCK_BYTES + 1;
            SushiBLAS::Detail::ScratchBlock* big = shelf_->take(above, 0);
            shelf_->give_back(big);

            SushiBLAS::Detail::ScratchBlock* again = shelf_->take(above, 0);

            EXPECT_EQ(shelf_->unpooled_leases(), std::uint64_t{2});
            EXPECT_EQ(shelf_->pool_misses(), std::uint64_t{0});
            shelf_->give_back(again);
        }

        /** @brief Runs one row of hit_rows() on an empty shelf. */
        class Small_ScratchShelfLease : public Small_ScratchShelf,
                                        public ::testing::WithParamInterface<HitRow>
        {
            protected:
                /** @brief Leaves one idle block of the smallest class on the shelf when asked to. */
                void return_an_idle_block_when(bool idle)
                {
                    if (idle)
                        shelf_->give_back(shelf_->take(128, 0));
                }
        };

        TEST_P(Small_ScratchShelfLease, CountsAHitOrAMissByWhetherItsClassHoldsAnIdleBlock)
        {
            return_an_idle_block_when(GetParam().idle_first);

            SushiBLAS::Detail::ScratchBlock* lease = shelf_->take(128, 0);

            EXPECT_EQ(shelf_->pool_misses(), std::uint64_t{1});
            EXPECT_EQ(shelf_->pool_hits(), GetParam().hits);
            shelf_->give_back(lease);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_ScratchShelfLease, ::testing::ValuesIn(hit_rows()),
            [](const ::testing::TestParamInfo<HitRow>& info)
            { return std::string(info.param.name); });
    } // namespace Test
} // namespace SushiBLAS
