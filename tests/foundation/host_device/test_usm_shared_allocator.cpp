/**************************************************************************/
/* test_usm_shared_allocator.cpp                                          */
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
 * @file test_usm_shared_allocator.cpp
 * @brief Tests that UsmSharedAllocator keeps the SharedAllocator contract on the queue of the shared test environment.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/host_device/usm_shared_allocator.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include "support/shared_allocator_contract.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The adapter a contract suite runs against, on the queue the test program owns.
        struct UsmSubject
        {
            /// @brief The allocator under test.
            UsmSharedAllocator allocator{TestEnvironment::queue()};
        };

        /** @brief Runs the SharedAllocator contract against a subject. */
        template<typename Subject>
        class Medium_SharedAllocatorContract : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        using UsmSubjects = ::testing::Types<UsmSubject>;
        SB_TYPED_TEST_SUITE(Medium_SharedAllocatorContract, UsmSubjects);

        TYPED_TEST(Medium_SharedAllocatorContract, ABlockIsAlignedAsRequested)
        {
            SharedAllocatorContract::expect_block_is_aligned(this->subject.allocator);
        }

        TYPED_TEST(Medium_SharedAllocatorContract, ABlockHoldsWhatIsWritten)
        {
            SharedAllocatorContract::expect_block_holds_what_is_written(this->subject.allocator);
        }

        TYPED_TEST(Medium_SharedAllocatorContract, LiveBlocksDoNotOverlap)
        {
            SharedAllocatorContract::expect_live_blocks_do_not_overlap(this->subject.allocator);
        }

        TYPED_TEST(Medium_SharedAllocatorContract, ARequestIsServedAgainAfterABlockIsTakenBack)
        {
            SharedAllocatorContract::expect_a_request_is_served_again_after_a_block_is_taken_back(
                this->subject.allocator);
        }
    } // namespace Test
} // namespace SushiBLAS
