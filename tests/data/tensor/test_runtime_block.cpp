/**************************************************************************/
/* test_runtime_block.cpp                                                 */
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
 * @file test_runtime_block.cpp
 * @brief Tests that RuntimeBlock allocates or borrows a buffer and copies through the buffer's range calls.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <utility>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/runtime_block.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include "support/owned_memory.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Holds the bytes of the memory the borrowing tests view.
        constexpr std::size_t K_BORROWED_BYTES = 64;

        TEST(Medium_RuntimeBlock, AnAllocatingBlockIsNotBorrowed)
        {
            const RuntimeBlock block(TestEnvironment::runtime(), K_BORROWED_BYTES);

            EXPECT_FALSE(block.is_borrowed());
        }

        TEST(Medium_RuntimeBlock, AnAllocatingBlockNamesNoOwner)
        {
            const RuntimeBlock block(TestEnvironment::runtime(), K_BORROWED_BYTES);

            EXPECT_FALSE(names_an_owner(block.owner_liveness()));
        }

        TEST(Medium_RuntimeBlock, ABlockReportsTheHandleOfItsBuffer)
        {
            const RuntimeBlock block(TestEnvironment::runtime(), K_BORROWED_BYTES);

            EXPECT_FALSE(block.global_handle().is_null());
        }

        TEST(Medium_RuntimeBlock, TwoLiveBlocksReportTwoHandles)
        {
            const RuntimeBlock first(TestEnvironment::runtime(), K_BORROWED_BYTES);
            const RuntimeBlock second(TestEnvironment::runtime(), K_BORROWED_BYTES);

            EXPECT_NE(first.global_handle(), second.global_handle());
        }

        TEST(Medium_RuntimeBlock, AStorageAndATensorOverTheBlockReportItsHandle)
        {
            auto block = std::make_unique<RuntimeBlock>(TestEnvironment::runtime(), K_BORROWED_BYTES);
            const SushiRuntime::GlobalHandle expected = block->global_handle();
            const SushiRuntime::sushi_ptr<Storage> storage =
                SushiRuntime::make_sushi<Storage>(std::move(block));
            const Tensor tensor(storage, {4});

            EXPECT_EQ(storage->global_handle(), expected);
            EXPECT_EQ(tensor.global_handle(), expected);
        }

        TEST(Medium_RuntimeBlock, ABorrowingBlockReportsTheHandleOfTheBufferItViews)
        {
            const OwnedMemory memory(K_BORROWED_BYTES);
            const RuntimeBlock block(RuntimeBlock::Borrowed{}, TestEnvironment::runtime(),
                                     memory.borrow(0, K_BORROWED_BYTES));

            EXPECT_FALSE(block.global_handle().is_null());
        }

        TEST(Medium_RuntimeBlock, ItsQueueIsTheQueueOfItsDevice)
        {
            const RuntimeBlock block(TestEnvironment::runtime(), K_BORROWED_BYTES);

            EXPECT_EQ(block.queue(), &TestEnvironment::queue(0));
        }

        TEST(Medium_RuntimeBlock, ABorrowingBlockViewsTheOwnersBytes)
        {
            const OwnedMemory memory(K_BORROWED_BYTES);
            const RuntimeBlock block(RuntimeBlock::Borrowed{}, TestEnvironment::runtime(),
                                     memory.borrow(0, K_BORROWED_BYTES));

            EXPECT_EQ(block.data(), static_cast<void*>(memory.data()));
        }

        TEST(Medium_RuntimeBlock, ABorrowingBlockKeepsTheSizeOfTheBufferUnrounded)
        {
            const OwnedMemory memory(K_BORROWED_BYTES);
            const RuntimeBlock block(RuntimeBlock::Borrowed{}, TestEnvironment::runtime(),
                                     memory.borrow(0, K_BORROWED_BYTES / 2));

            EXPECT_EQ(block.size_bytes(), K_BORROWED_BYTES / 2);
        }

        TEST(Medium_RuntimeBlock, ABorrowingBlockIsBorrowed)
        {
            const OwnedMemory memory(K_BORROWED_BYTES);
            const RuntimeBlock block(RuntimeBlock::Borrowed{}, TestEnvironment::runtime(),
                                     memory.borrow(0, K_BORROWED_BYTES));

            EXPECT_TRUE(block.is_borrowed());
        }

        TEST(Medium_RuntimeBlock, ABorrowingBlockWatchesALiveOwner)
        {
            const OwnedMemory memory(K_BORROWED_BYTES);
            const RuntimeBlock block(RuntimeBlock::Borrowed{}, TestEnvironment::runtime(),
                                     memory.borrow(0, K_BORROWED_BYTES));

            EXPECT_FALSE(block.owner_liveness().expired());
        }

        TEST(Medium_RuntimeBlock, ABorrowingBlockSeesTheOwnerLetGo)
        {
            OwnedMemory memory(K_BORROWED_BYTES);
            const RuntimeBlock block(RuntimeBlock::Borrowed{}, TestEnvironment::runtime(),
                                     memory.borrow(0, K_BORROWED_BYTES));

            memory.release();

            EXPECT_TRUE(block.owner_liveness().expired());
        }

        TEST(Medium_RuntimeBlock, ABorrowingBlockRefusesAMisalignedBuffer)
        {
            const OwnedMemory memory(K_BORROWED_BYTES);

            EXPECT_THROW(RuntimeBlock(RuntimeBlock::Borrowed{}, TestEnvironment::runtime(),
                                      memory.borrow(1, K_BORROWED_BYTES / 2)),
                         std::runtime_error);
        }

        TEST(Medium_RuntimeBlock, ACopyOutOfABorrowedBlockReadsTheOwnersBytes)
        {
            const OwnedMemory memory(K_BORROWED_BYTES);
            memory.data()[5] = std::byte{9};
            const RuntimeBlock block(RuntimeBlock::Borrowed{}, TestEnvironment::runtime(),
                                     memory.borrow(0, K_BORROWED_BYTES));
            std::vector<std::byte> read(2);

            block.copy_to_host(read.data(), 4, read.size());

            EXPECT_EQ(read[1], std::byte{9});
        }

        TEST(Medium_RuntimeBlock, ADeviceBlockRoundTripsBytesThroughTheHostCopies)
        {
            const RuntimeBlock block(TestEnvironment::runtime(), K_BORROWED_BYTES,
                                     SushiRuntime::Residency::Device);
            std::vector<std::byte> written(K_BORROWED_BYTES / 2);
            for (std::size_t i = 0; i < written.size(); ++i)
                written[i] = static_cast<std::byte>(i + 1);
            std::vector<std::byte> read(written.size());

            block.copy_from_host(written.data(), 8, written.size());
            block.copy_to_host(read.data(), 8, read.size());

            EXPECT_EQ(read, written);
        }

        TEST(Medium_RuntimeBlock, ADeviceBlockCopyRefusesARangePastItsEnd)
        {
            const RuntimeBlock block(TestEnvironment::runtime(), K_BORROWED_BYTES,
                                     SushiRuntime::Residency::Device);
            std::vector<std::byte> host(K_BORROWED_BYTES);

            EXPECT_THROW(block.copy_to_host(host.data(), 1, block.size_bytes()),
                         std::runtime_error);
            EXPECT_THROW(block.copy_from_host(host.data(), 1, block.size_bytes()),
                         std::runtime_error);
        }
    } // namespace Test
} // namespace SushiBLAS
