/**************************************************************************/
/* test_storage_block_contract.cpp                                        */
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
 * @file test_storage_block_contract.cpp
 * @brief Tests that the host and runtime adapters of StorageBlock keep the StorageBlock contract.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/tensor/storage_block.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include "support/storage_block_contract.hpp"
#include "support/tensor_subjects.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the StorageBlock contract against an adapter that needs no runtime. */
        template<typename Subject>
        class Small_StorageBlockContract : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        /** @brief Runs the StorageBlock contract against an adapter over the shared runtime. */
        template<typename Subject>
        class Medium_StorageBlockContract : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        using HostSubjects = ::testing::Types<HostBlockSubject>;
        using RuntimeSubjects = ::testing::Types<RuntimeBlockSubject>;
        SB_TYPED_TEST_SUITE(Small_StorageBlockContract, HostSubjects);
        SB_TYPED_TEST_SUITE(Medium_StorageBlockContract, RuntimeSubjects);

        TYPED_TEST(Small_StorageBlockContract, TheSizeRoundsUpToTheAlignment)
        {
            StorageBlockContract::expect_size_rounds_up_to_the_alignment(this->subject);
        }

        TYPED_TEST(Small_StorageBlockContract, TheBaseIsAligned)
        {
            StorageBlockContract::expect_base_is_aligned(this->subject);
        }

        TYPED_TEST(Small_StorageBlockContract, TheBlockHoldsWhatIsWritten)
        {
            StorageBlockContract::expect_block_holds_what_is_written(this->subject);
        }

        TYPED_TEST(Small_StorageBlockContract, CopiesRoundTripAtAnOffset)
        {
            StorageBlockContract::expect_copies_round_trip_at_an_offset(this->subject);
        }

        TYPED_TEST(Small_StorageBlockContract, CopiesReachTheBytesDataAddresses)
        {
            StorageBlockContract::expect_copies_reach_the_bytes_data_addresses(this->subject);
        }

        TYPED_TEST(Small_StorageBlockContract, ACopyPastTheEndIsRefused)
        {
            StorageBlockContract::expect_copy_past_the_end_is_refused(this->subject);
        }

        TYPED_TEST(Small_StorageBlockContract, ItReportsItsResidencyAndDevice)
        {
            StorageBlockContract::expect_reports_its_residency_and_device(this->subject);
        }

        TYPED_TEST(Small_StorageBlockContract, ItHasAQueueExactlyWhenADeviceOwnsIt)
        {
            StorageBlockContract::expect_queue_follows_device_ownership(this->subject);
        }

        TYPED_TEST(Small_StorageBlockContract, AnOwnedBlockNamesNoOwner)
        {
            StorageBlockContract::expect_an_owned_block_names_no_owner(this->subject);
        }

        TYPED_TEST(Small_StorageBlockContract, AnEmptyRequestIsRefused)
        {
            StorageBlockContract::expect_empty_request_is_refused(this->subject);
        }

        TYPED_TEST(Small_StorageBlockContract, AnOverflowingRequestIsRefused)
        {
            StorageBlockContract::expect_overflowing_request_is_refused(this->subject);
        }

        TYPED_TEST(Small_StorageBlockContract, LiveBlocksDoNotOverlap)
        {
            StorageBlockContract::expect_live_blocks_do_not_overlap(this->subject);
        }

        TYPED_TEST(Medium_StorageBlockContract, TheSizeRoundsUpToTheAlignment)
        {
            StorageBlockContract::expect_size_rounds_up_to_the_alignment(this->subject);
        }

        TYPED_TEST(Medium_StorageBlockContract, TheBaseIsAligned)
        {
            StorageBlockContract::expect_base_is_aligned(this->subject);
        }

        TYPED_TEST(Medium_StorageBlockContract, TheBlockHoldsWhatIsWritten)
        {
            StorageBlockContract::expect_block_holds_what_is_written(this->subject);
        }

        TYPED_TEST(Medium_StorageBlockContract, CopiesRoundTripAtAnOffset)
        {
            StorageBlockContract::expect_copies_round_trip_at_an_offset(this->subject);
        }

        TYPED_TEST(Medium_StorageBlockContract, CopiesReachTheBytesDataAddresses)
        {
            StorageBlockContract::expect_copies_reach_the_bytes_data_addresses(this->subject);
        }

        TYPED_TEST(Medium_StorageBlockContract, ACopyPastTheEndIsRefused)
        {
            StorageBlockContract::expect_copy_past_the_end_is_refused(this->subject);
        }

        TYPED_TEST(Medium_StorageBlockContract, ItReportsItsResidencyAndDevice)
        {
            StorageBlockContract::expect_reports_its_residency_and_device(this->subject);
        }

        TYPED_TEST(Medium_StorageBlockContract, ItHasAQueueExactlyWhenADeviceOwnsIt)
        {
            StorageBlockContract::expect_queue_follows_device_ownership(this->subject);
        }

        TYPED_TEST(Medium_StorageBlockContract, AnOwnedBlockNamesNoOwner)
        {
            StorageBlockContract::expect_an_owned_block_names_no_owner(this->subject);
        }

        TYPED_TEST(Medium_StorageBlockContract, AnEmptyRequestIsRefused)
        {
            StorageBlockContract::expect_empty_request_is_refused(this->subject);
        }

        TYPED_TEST(Medium_StorageBlockContract, AnOverflowingRequestIsRefused)
        {
            StorageBlockContract::expect_overflowing_request_is_refused(this->subject);
        }

        TYPED_TEST(Medium_StorageBlockContract, LiveBlocksDoNotOverlap)
        {
            StorageBlockContract::expect_live_blocks_do_not_overlap(this->subject);
        }
    } // namespace Test
} // namespace SushiBLAS
