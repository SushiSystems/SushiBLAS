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
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
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
            return SushiRuntime::make_sushi<Storage>(TestEnvironment::runtime(), n_bytes);
        }

        /// @brief Returns a storage over count bytes of memory, from offset.
        static SushiRuntime::sushi_ptr<Storage> borrowed_of(const OwnedMemory& memory,
                                                           std::size_t offset, std::size_t count)
        {
            return SushiRuntime::make_sushi<Storage>(Storage::Borrowed{},
                                                     TestEnvironment::runtime(),
                                                     memory.borrow(offset, count));
        }

        TEST(Medium_Storage, RoundsItsSizeUpToTheAlignment)
        {
            EXPECT_EQ(storage_of(1)->size_bytes(), K_STORAGE_ALIGNMENT);
            EXPECT_EQ(storage_of(K_STORAGE_ALIGNMENT)->size_bytes(), K_STORAGE_ALIGNMENT);
            EXPECT_EQ(storage_of(K_STORAGE_ALIGNMENT + 1)->size_bytes(), 2 * K_STORAGE_ALIGNMENT);
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

        TEST(Medium_Storage, RefusesAnEmptyRequestAndADeviceItDoesNotHave)
        {
            EXPECT_THROW((void)storage_of(0), std::runtime_error);
            EXPECT_THROW((void)SushiRuntime::make_sushi<Storage>(
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

        TEST(Medium_Storage, ABorrowedStorageViewsTheOwnersMemoryWithoutAllocating)
        {
            const OwnedMemory memory(256);
            const std::size_t before = TestEnvironment::live_allocation_count();

            const auto storage = borrowed_of(memory, 0, 200);

            EXPECT_EQ(TestEnvironment::live_allocation_count(), before);
            EXPECT_TRUE(storage->is_borrowed());
            EXPECT_EQ(storage->allocation_base(), static_cast<void*>(memory.data()));
            EXPECT_EQ(storage->size_bytes(), std::size_t{200});
            EXPECT_EQ(storage->residency(), SushiRuntime::Residency::Shared);
            EXPECT_EQ(storage->device_index(), std::size_t{0});
            EXPECT_EQ(&storage->queue(), &TestEnvironment::queue(0));
        }

        TEST(Medium_Storage, AnOwnedStorageIsNotBorrowedAndWatchesNoOwner)
        {
            const auto storage = storage_of(64);

            EXPECT_FALSE(storage->is_borrowed());
            EXPECT_TRUE(storage->owner_liveness().expired());
        }

        TEST(Medium_Storage, ABorrowedStorageWatchesTheOwnerAndLeavesItsMemoryAlone)
        {
            OwnedMemory memory(64);
            memory.data()[0] = std::byte{7};
            {
                const auto storage = borrowed_of(memory, 0, 64);
                EXPECT_FALSE(storage->owner_liveness().expired());

                memory.release();
                EXPECT_TRUE(storage->owner_liveness().expired());
            }

            EXPECT_EQ(memory.data()[0], std::byte{7});
        }

        TEST(Medium_Storage, ABorrowedStorageRefusesAnEmptyOrMisalignedBuffer)
        {
            const OwnedMemory memory(64);

            EXPECT_THROW((void)borrowed_of(memory, 0, 0), std::runtime_error);
            EXPECT_THROW((void)borrowed_of(memory, 1, 32), std::runtime_error);
        }

        TEST(Medium_Storage, ATensorOverABorrowedStorageIsHeldToTheBufferSize)
        {
            const OwnedMemory memory(64);
            const auto storage = borrowed_of(memory, 0, 64);

            const Tensor fits(storage, {16}, Core::DataType::Float32);
            EXPECT_EQ(fits.region().base, static_cast<void*>(memory.data()));
            EXPECT_THROW((void)Tensor(storage, {17}, Core::DataType::Float32), std::runtime_error);
        }
    } // namespace Test
} // namespace SushiBLAS
