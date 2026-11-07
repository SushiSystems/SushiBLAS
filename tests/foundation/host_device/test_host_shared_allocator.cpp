/**************************************************************************/
/* test_host_shared_allocator.cpp                                         */
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
 * @file test_host_shared_allocator.cpp
 * @brief Tests that HostSharedAllocator keeps the SharedAllocator contract and counts, poisons and refuses on request.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/host_device/host_shared_allocator.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include "support/shared_allocator_contract.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The adapter a contract suite runs against.
        struct HostSubject
        {
            /// @brief The allocator under test.
            HostSharedAllocator allocator;
        };

        /** @brief Runs the SharedAllocator contract against a subject. */
        template<typename Subject>
        class Small_SharedAllocatorContract : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        using HostSubjects = ::testing::Types<HostSubject>;
        SB_TYPED_TEST_SUITE(Small_SharedAllocatorContract, HostSubjects);

        TYPED_TEST(Small_SharedAllocatorContract, ABlockIsAlignedAsRequested)
        {
            SharedAllocatorContract::expect_block_is_aligned(this->subject.allocator);
        }

        TYPED_TEST(Small_SharedAllocatorContract, ABlockHoldsWhatIsWritten)
        {
            SharedAllocatorContract::expect_block_holds_what_is_written(this->subject.allocator);
        }

        TYPED_TEST(Small_SharedAllocatorContract, LiveBlocksDoNotOverlap)
        {
            SharedAllocatorContract::expect_live_blocks_do_not_overlap(this->subject.allocator);
        }

        TYPED_TEST(Small_SharedAllocatorContract, ARequestIsServedAgainAfterABlockIsTakenBack)
        {
            SharedAllocatorContract::expect_a_request_is_served_again_after_a_block_is_taken_back(
                this->subject.allocator);
        }

        TEST(Small_HostSharedAllocator, CountsTheBlocksThatWentOutAndCameBack)
        {
            HostSharedAllocator allocator;

            void* first = allocator.allocate(64, 64);
            void* second = allocator.allocate(64, 64);
            allocator.deallocate(first);

            EXPECT_EQ(allocator.allocation_count(), std::size_t{2});
            EXPECT_EQ(allocator.free_count(), std::size_t{1});
            allocator.deallocate(second);
        }

        TEST(Small_HostSharedAllocator, RemembersTheLastRequest)
        {
            HostSharedAllocator allocator;

            allocator.deallocate(allocator.allocate(96, 32));

            EXPECT_EQ(allocator.last_bytes(), std::size_t{96});
            EXPECT_EQ(allocator.last_alignment(), std::size_t{32});
        }

        TEST(Small_HostSharedAllocator, StartsEveryBlockWithThePoisonByte)
        {
            HostSharedAllocator allocator;
            void* block = allocator.allocate(48, 16);

            const auto* bytes = static_cast<const unsigned char*>(block);

            EXPECT_EQ(std::vector<unsigned char>(bytes, bytes + 48),
                      std::vector<unsigned char>(48, K_HOST_BLOCK_POISON));
            allocator.deallocate(block);
        }

        TEST(Small_HostSharedAllocator, AFailingAllocatorReturnsNullAndCountsNoBlock)
        {
            HostSharedAllocator allocator;
            allocator.set_failing(true);

            void* block = allocator.allocate(64, 64);

            EXPECT_EQ(block, nullptr);
            EXPECT_EQ(allocator.allocation_count(), std::size_t{0});
            EXPECT_EQ(allocator.last_bytes(), std::size_t{64});
        }

        TEST(Small_HostSharedAllocator, ABlockItDidNotServeIsIgnored)
        {
            HostSharedAllocator allocator;
            int foreign = 0;

            allocator.deallocate(&foreign);

            EXPECT_EQ(allocator.free_count(), std::size_t{0});
        }
    } // namespace Test
} // namespace SushiBLAS
