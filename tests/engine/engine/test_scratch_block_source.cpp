/**************************************************************************/
/* test_scratch_block_source.cpp                                          */
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
 * @file test_scratch_block_source.cpp
 * @brief Tests that the host and runtime adapters of ScratchBlockSource keep the BlockSource
 *        contract, and what the host adapter counts.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/engine/host_scratch_source.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include "support/scratch_block_source_contract.hpp"
#include "support/scratch_source_subjects.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the BlockSource contract against a subject on the host. */
        template<typename Subject>
        class Small_BlockSourceContract : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        /** @brief Runs the BlockSource contract against a subject on the shared runtime. */
        template<typename Subject>
        class Medium_BlockSourceContract : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        using HostSourceSubjects = ::testing::Types<HostSourceSubject>;
        using RuntimeSourceSubjects = ::testing::Types<RuntimeSourceSubject>;
        SB_TYPED_TEST_SUITE(Small_BlockSourceContract, HostSourceSubjects);
        SB_TYPED_TEST_SUITE(Medium_BlockSourceContract, RuntimeSourceSubjects);

        TYPED_TEST(Small_BlockSourceContract, ABlockHoldsTheRequest)
        {
            BlockSourceContract::expect_block_holds_the_request(this->subject);
        }

        TYPED_TEST(Small_BlockSourceContract, TheBaseIsAligned)
        {
            BlockSourceContract::expect_base_is_aligned(this->subject);
        }

        TYPED_TEST(Small_BlockSourceContract, LiveBlocksDoNotShareAnAddress)
        {
            BlockSourceContract::expect_live_blocks_do_not_share_an_address(this->subject);
        }

        TYPED_TEST(Small_BlockSourceContract, ItServesAtLeastOneDevice)
        {
            BlockSourceContract::expect_serves_at_least_one_device(this->subject);
        }

        TYPED_TEST(Small_BlockSourceContract, AnUnknownDeviceIsRefused)
        {
            BlockSourceContract::expect_unknown_device_is_refused(this->subject);
        }

        TYPED_TEST(Small_BlockSourceContract, ZeroBytesAreRefused)
        {
            BlockSourceContract::expect_zero_bytes_are_refused(this->subject);
        }

        TYPED_TEST(Medium_BlockSourceContract, ABlockHoldsTheRequest)
        {
            BlockSourceContract::expect_block_holds_the_request(this->subject);
        }

        TYPED_TEST(Medium_BlockSourceContract, TheBaseIsAligned)
        {
            BlockSourceContract::expect_base_is_aligned(this->subject);
        }

        TYPED_TEST(Medium_BlockSourceContract, LiveBlocksDoNotShareAnAddress)
        {
            BlockSourceContract::expect_live_blocks_do_not_share_an_address(this->subject);
        }

        TYPED_TEST(Medium_BlockSourceContract, ItServesAtLeastOneDevice)
        {
            BlockSourceContract::expect_serves_at_least_one_device(this->subject);
        }

        TYPED_TEST(Medium_BlockSourceContract, AnUnknownDeviceIsRefused)
        {
            BlockSourceContract::expect_unknown_device_is_refused(this->subject);
        }

        TYPED_TEST(Medium_BlockSourceContract, ZeroBytesAreRefused)
        {
            BlockSourceContract::expect_zero_bytes_are_refused(this->subject);
        }

        /** @brief Tests that a lease takes its block from the source of the pool. */
        TEST(Small_ScratchPoolOverASource, ALeaseTakesItsBlockFromTheSource)
        {
            HostScratchSource source;
            ScratchPool pool(source);

            const ScratchBuffer lease(pool, 200, 0);

            EXPECT_EQ(source.served_blocks(), std::size_t{1});
            EXPECT_EQ(lease.storage()->size_bytes(), 2 * K_STORAGE_ALIGNMENT);
        }

        /** @brief Tests that a block the pool took back is lent again without asking the source. */
        TEST(Small_ScratchPoolOverASource, AReturnedBlockIsLentAgainWithoutTheSource)
        {
            HostScratchSource source;
            ScratchPool pool(source);
            {
                const ScratchBuffer first(pool, 200, 0);
            }

            const ScratchBuffer second(pool, 200, 0);

            EXPECT_EQ(source.served_blocks(), std::size_t{1});
        }
    } // namespace Test
} // namespace SushiBLAS
