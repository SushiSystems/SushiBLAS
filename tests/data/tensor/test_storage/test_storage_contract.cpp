/**************************************************************************/
/* test_storage_contract.cpp                                              */
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
 * @file test_storage_contract.cpp
 * @brief Tests that the host and runtime adapters of Storage keep the Storage contract, and that a
 *        host storage has no queue.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include "support/storage_contract.hpp"
#include "support/tensor_subjects.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the Storage contract against a subject on the host. */
        template<typename Subject>
        class Small_StorageContract : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        /** @brief Runs the Storage contract against a subject on the shared runtime. */
        template<typename Subject>
        class Medium_StorageContract : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        using HostSubjects = ::testing::Types<HostTensorSubject>;
        using RuntimeSubjects = ::testing::Types<RuntimeTensorSubject>;
        SB_TYPED_TEST_SUITE(Small_StorageContract, HostSubjects);
        SB_TYPED_TEST_SUITE(Medium_StorageContract, RuntimeSubjects);

        TYPED_TEST(Small_StorageContract, TheSizeRoundsUpToTheAlignment)
        {
            StorageContract::expect_size_rounds_up_to_the_alignment(this->subject);
        }

        TYPED_TEST(Small_StorageContract, TheBaseIsAligned)
        {
            StorageContract::expect_base_is_aligned(this->subject);
        }

        TYPED_TEST(Small_StorageContract, TheAllocationHoldsWhatIsWritten)
        {
            StorageContract::expect_allocation_holds_what_is_written(this->subject);
        }

        TYPED_TEST(Small_StorageContract, ATensorsDataIsTheBase)
        {
            StorageContract::expect_tensor_data_is_the_base(this->subject);
        }

        TYPED_TEST(Small_StorageContract, ItReportsItsResidencyAndDevice)
        {
            StorageContract::expect_reports_its_residency_and_device(this->subject);
        }

        TYPED_TEST(Small_StorageContract, AnEmptyRequestIsRefused)
        {
            StorageContract::expect_empty_request_is_refused(this->subject);
        }

        TYPED_TEST(Small_StorageContract, AnOverflowingRequestIsRefused)
        {
            StorageContract::expect_overflowing_request_is_refused(this->subject);
        }

        TYPED_TEST(Small_StorageContract, LiveStoragesDoNotOverlap)
        {
            StorageContract::expect_live_storages_do_not_overlap(this->subject);
        }

        TYPED_TEST(Medium_StorageContract, TheSizeRoundsUpToTheAlignment)
        {
            StorageContract::expect_size_rounds_up_to_the_alignment(this->subject);
        }

        TYPED_TEST(Medium_StorageContract, TheBaseIsAligned)
        {
            StorageContract::expect_base_is_aligned(this->subject);
        }

        TYPED_TEST(Medium_StorageContract, TheAllocationHoldsWhatIsWritten)
        {
            StorageContract::expect_allocation_holds_what_is_written(this->subject);
        }

        TYPED_TEST(Medium_StorageContract, ATensorsDataIsTheBase)
        {
            StorageContract::expect_tensor_data_is_the_base(this->subject);
        }

        TYPED_TEST(Medium_StorageContract, ItReportsItsResidencyAndDevice)
        {
            StorageContract::expect_reports_its_residency_and_device(this->subject);
        }

        TYPED_TEST(Medium_StorageContract, AnEmptyRequestIsRefused)
        {
            StorageContract::expect_empty_request_is_refused(this->subject);
        }

        TYPED_TEST(Medium_StorageContract, AnOverflowingRequestIsRefused)
        {
            StorageContract::expect_overflowing_request_is_refused(this->subject);
        }

        TYPED_TEST(Medium_StorageContract, LiveStoragesDoNotOverlap)
        {
            StorageContract::expect_live_storages_do_not_overlap(this->subject);
        }

        TEST(Small_HostStorage, ItHasNoQueue)
        {
            const auto storage = SushiRuntime::make_sushi<Storage>(Storage::Host{}, 64);

            EXPECT_THROW((void)storage->queue(), UnsupportedExecution);
        }

        TEST(Small_HostStorage, ItIsHostMemoryAndNotBorrowed)
        {
            const auto storage = SushiRuntime::make_sushi<Storage>(Storage::Host{}, 64);

            EXPECT_TRUE(storage->is_host());
            EXPECT_FALSE(storage->is_borrowed());
        }

        TEST(Medium_Storage, ARuntimeStorageIsNotHostMemory)
        {
            const auto storage = SushiRuntime::make_sushi<Storage>(TestEnvironment::runtime(), 64);

            EXPECT_FALSE(storage->is_host());
        }
    } // namespace Test
} // namespace SushiBLAS
