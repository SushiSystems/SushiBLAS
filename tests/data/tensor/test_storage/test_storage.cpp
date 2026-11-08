/**************************************************************************/
/* test_storage.cpp                                                       */
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
 * @file test_storage.cpp
 * @brief Tests that Storage rounds its size up, aligns its allocation, refuses a bad request and
 *        views a buffer it borrows.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/tensor/host_block.hpp>
#include <SushiBLAS/tensor/runtime_block.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include "support/owned_memory.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Returns a storage of n_bytes on the shared runtime's first device.
        static SushiRuntime::sushi_ptr<Storage> storage_of(std::size_t n_bytes)
        {
            return make_runtime_storage(TestEnvironment::runtime(), n_bytes);
        }

        /// @brief Returns a storage over count bytes of memory, from offset.
        static SushiRuntime::sushi_ptr<Storage> borrowed_of(const OwnedMemory& memory,
                                                           std::size_t offset, std::size_t count)
        {
            return borrow_runtime_storage(TestEnvironment::runtime(),
                                          memory.borrow(offset, count));
        }

        /// @brief A requested size and the size a storage reports for it.
        struct RoundingRow
        {
            std::size_t requested;
            std::size_t reported;
        };

        class Small_StorageRounding : public ::testing::TestWithParam<RoundingRow>
        {
        };

        TEST_P(Small_StorageRounding, ReportsTheRequestRoundedUpToTheAlignment)
        {
            const RoundingRow row = GetParam();

            EXPECT_EQ(make_host_storage(row.requested)->size_bytes(), row.reported);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_StorageRounding,
            ::testing::Values(RoundingRow{1, K_STORAGE_ALIGNMENT},
                              RoundingRow{K_STORAGE_ALIGNMENT, K_STORAGE_ALIGNMENT},
                              RoundingRow{K_STORAGE_ALIGNMENT + 1, 2 * K_STORAGE_ALIGNMENT}));

        TEST(Small_Storage, ForwardsEveryQuestionToItsBlock)
        {
            const auto storage = make_host_storage(64);

            EXPECT_EQ(storage->allocation_base(), storage->block().data());
            EXPECT_EQ(storage->size_bytes(), storage->block().size_bytes());
            EXPECT_EQ(storage->residency(), storage->block().residency());
            EXPECT_EQ(storage->device_index(), storage->block().device_index());
        }

        TEST(Small_Storage, AStorageWithNoDeviceOwnerHasNoQueue)
        {
            const auto storage = make_host_storage(64);

            EXPECT_EQ(storage->block().queue(), nullptr);
            EXPECT_THROW((void)storage->queue(), UnsupportedExecution);
        }

        TEST(Small_Storage, CopiesBytesThroughItsBlock)
        {
            const auto storage = make_host_storage(64);
            const std::byte written[3] = {std::byte{1}, std::byte{2}, std::byte{3}};
            std::byte read[3] = {};

            storage->copy_from_host(written, 8, sizeof(written));
            storage->copy_to_host(read, 8, sizeof(read));

            EXPECT_EQ(read[0], written[0]);
            EXPECT_EQ(read[2], written[2]);
        }

        TEST(Small_Storage, RefusesAMissingBlock)
        {
            EXPECT_THROW((void)SushiRuntime::make_sushi<Storage>(std::unique_ptr<StorageBlock>()),
                         std::runtime_error);
        }

        TEST(Small_Storage, ATensorsDataIsTheBaseOfAHostStorage)
        {
            const auto storage = make_host_storage(64);
            const Tensor tensor(storage, {4, 4}, Core::DataType::Float32);

            EXPECT_EQ(tensor.data(), storage->allocation_base());
        }

        TEST(Medium_Storage, ItsAllocationIsAligned)
        {
            const auto storage = storage_of(1000);

            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(storage->allocation_base()) %
                          K_STORAGE_ALIGNMENT,
                      std::uintptr_t{0});
        }

        TEST(Medium_Storage, ReportsItsResidencyDeviceAndQueue)
        {
            const auto storage = storage_of(64);

            EXPECT_EQ(storage->residency(), SushiRuntime::Residency::Shared);
            EXPECT_EQ(storage->device_index(), std::size_t{0});
            EXPECT_EQ(&storage->queue(), &TestEnvironment::queue(0));
        }

        TEST(Medium_Storage, RefusesAnEmptyRequest)
        {
            EXPECT_THROW((void)storage_of(0), std::runtime_error);
        }

        TEST(Medium_Storage, RefusesADeviceItDoesNotHave)
        {
            EXPECT_THROW((void)make_runtime_storage(
                             TestEnvironment::runtime(), 64, SushiRuntime::Residency::Shared,
                             TestEnvironment::device_count()),
                         std::runtime_error);
        }

        TEST(Medium_Storage, ReturnsItsAllocationWhenTheLastHandleGoes)
        {
            const std::size_t before = TestEnvironment::live_allocation_count();
            {
                const auto storage = storage_of(256);
                EXPECT_EQ(TestEnvironment::live_allocation_count(), before + 1);
            }

            EXPECT_EQ(TestEnvironment::live_allocation_count(), before);
        }

        TEST(Medium_Storage, ABorrowedStorageAllocatesNothing)
        {
            const OwnedMemory memory(256);
            const std::size_t before = TestEnvironment::live_allocation_count();

            const auto storage = borrowed_of(memory, 0, 200);

            EXPECT_EQ(TestEnvironment::live_allocation_count(), before);
        }

        TEST(Medium_Storage, ABorrowedStorageViewsTheOwnersMemory)
        {
            const OwnedMemory memory(256);

            const auto storage = borrowed_of(memory, 0, 200);

            EXPECT_EQ(storage->allocation_base(), static_cast<void*>(memory.data()));
            EXPECT_EQ(storage->size_bytes(), std::size_t{200});
        }

        TEST(Medium_Storage, ABorrowedStorageReportsTheResidencyDeviceAndQueueOfTheBuffer)
        {
            const OwnedMemory memory(256);

            const auto storage = borrowed_of(memory, 0, 200);

            EXPECT_EQ(storage->residency(), SushiRuntime::Residency::Shared);
            EXPECT_EQ(storage->device_index(), std::size_t{0});
            EXPECT_EQ(&storage->queue(), &TestEnvironment::queue(0));
        }

        TEST(Medium_Storage, AnOwnedStorageIsNotBorrowed)
        {
            const auto storage = storage_of(64);

            EXPECT_FALSE(names_an_owner(storage->owner_liveness()));
        }

        TEST(Medium_Storage, AnOwnedStorageWatchesNoOwner)
        {
            const auto storage = storage_of(64);

            EXPECT_TRUE(storage->owner_liveness().expired());
        }

        TEST(Medium_Storage, ABorrowedStorageNamesItsOwner)
        {
            const OwnedMemory memory(64);
            const auto storage = borrowed_of(memory, 0, 64);

            EXPECT_TRUE(names_an_owner(storage->owner_liveness()));
        }

        TEST(Medium_Storage, ABorrowedStorageSeesTheOwnerLetGo)
        {
            OwnedMemory memory(64);
            const auto storage = borrowed_of(memory, 0, 64);
            EXPECT_FALSE(storage->owner_liveness().expired());

            memory.release();

            EXPECT_TRUE(storage->owner_liveness().expired());
        }

        TEST(Medium_Storage, ABorrowedStorageLeavesTheOwnersMemoryAlone)
        {
            OwnedMemory memory(64);
            memory.data()[0] = std::byte{7};

            {
                const auto storage = borrowed_of(memory, 0, 64);
            }

            EXPECT_EQ(memory.data()[0], std::byte{7});
        }

        /// @brief A buffer a borrowed storage refuses: where it starts and how long it is.
        struct BadBufferRow
        {
            std::size_t offset;
            std::size_t count;
        };

        class Medium_StorageBorrowRefusal : public ::testing::TestWithParam<BadBufferRow>
        {
        };

        TEST_P(Medium_StorageBorrowRefusal, ABorrowedStorageRefusesTheBuffer)
        {
            const OwnedMemory memory(64);
            const BadBufferRow row = GetParam();

            EXPECT_THROW((void)borrowed_of(memory, row.offset, row.count), std::runtime_error);
        }

        // An empty buffer, and a buffer that starts one byte off the alignment.
        INSTANTIATE_TEST_SUITE_P(, Medium_StorageBorrowRefusal,
                                 ::testing::Values(BadBufferRow{0, 0}, BadBufferRow{1, 32}));
    } // namespace Test
} // namespace SushiBLAS
