/**************************************************************************/
/* test_storage_block.cpp                                                 */
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
 * @file test_storage_block.cpp
 * @brief Tests the defaults and helpers the StorageBlock port defines for every adapter.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/host_block.hpp>
#include <SushiBLAS/tensor/storage_block.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Holds the bytes of the block the range tests copy in and out of.
        constexpr std::size_t K_RANGE_BLOCK_BYTES = 16;

        TEST(Small_StorageBlock, ABlockThatOwnsItsMemoryNamesNoOwner)
        {
            const HostBlock block(K_RANGE_BLOCK_BYTES, K_STORAGE_ALIGNMENT);

            EXPECT_FALSE(names_an_owner(block.owner_liveness()));
        }

        TEST(Small_StorageBlock, AnObserverMadeFromAnOwnerNamesIt)
        {
            const SushiRuntime::LivenessToken owner = SushiRuntime::make_liveness();
            const SushiRuntime::LivenessObserver observer = owner;

            EXPECT_TRUE(names_an_owner(observer));
        }

        TEST(Small_StorageBlock, AnObserverStillNamesAnOwnerThatLetGo)
        {
            SushiRuntime::LivenessToken owner = SushiRuntime::make_liveness();
            const SushiRuntime::LivenessObserver observer = owner;

            owner.reset();

            EXPECT_TRUE(names_an_owner(observer));
        }

        TEST(Small_StorageBlock, ACopyOfTheWholeBlockReturnsEveryByteWritten)
        {
            const HostBlock block(K_RANGE_BLOCK_BYTES, K_STORAGE_ALIGNMENT);
            const std::vector<std::byte> written(K_RANGE_BLOCK_BYTES, std::byte{3});
            std::vector<std::byte> read(K_RANGE_BLOCK_BYTES);

            block.copy_from_host(written.data(), 0, written.size());
            block.copy_to_host(read.data(), 0, read.size());

            EXPECT_EQ(read, written);
        }

        TEST(Small_StorageBlock, ACopyOfOneByteMoreThanTheBlockHoldsIsRefused)
        {
            const HostBlock block(K_RANGE_BLOCK_BYTES, K_STORAGE_ALIGNMENT);
            std::byte read[K_RANGE_BLOCK_BYTES + 1] = {};

            EXPECT_THROW(block.copy_to_host(read, 0, K_RANGE_BLOCK_BYTES + 1), std::runtime_error);
        }

        TEST(Small_StorageBlock, AnEmptyCopyAtTheEndOfTheBlockLeavesTheDestinationAlone)
        {
            const HostBlock block(K_RANGE_BLOCK_BYTES, K_STORAGE_ALIGNMENT);
            std::byte read = std::byte{7};

            block.copy_to_host(&read, K_RANGE_BLOCK_BYTES, 0);

            EXPECT_EQ(read, std::byte{7});
        }

        TEST(Small_StorageBlock, AnOffsetPastTheEndOfTheBlockIsRefused)
        {
            const HostBlock block(K_RANGE_BLOCK_BYTES, K_STORAGE_ALIGNMENT);

            EXPECT_THROW(block.copy_from_host(nullptr, K_RANGE_BLOCK_BYTES + 1, 0),
                         std::runtime_error);
        }

        TEST(Small_StorageBlock, AnOffsetPlusSizeThatWrapsAroundIsRefused)
        {
            const HostBlock block(K_RANGE_BLOCK_BYTES, K_STORAGE_ALIGNMENT);

            EXPECT_THROW(block.copy_to_host(nullptr, 8, SIZE_MAX), std::runtime_error);
        }

        TEST(Small_RoundedStorageSize, RoundsUpToTheAlignment)
        {
            EXPECT_EQ(rounded_storage_size(K_STORAGE_ALIGNMENT + 1), 2 * K_STORAGE_ALIGNMENT);
        }

        TEST(Small_RoundedStorageSize, RefusesZero)
        {
            EXPECT_THROW((void)rounded_storage_size(0), std::runtime_error);
        }

        TEST(Small_RoundedStorageSize, RefusesASizeThatOverflowsWhenRounded)
        {
            EXPECT_THROW((void)rounded_storage_size(SIZE_MAX), std::runtime_error);
        }
    } // namespace Test
} // namespace SushiBLAS
