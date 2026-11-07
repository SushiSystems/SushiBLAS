/**************************************************************************/
/* test_host_block.cpp                                                    */
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
 * @file test_host_block.cpp
 * @brief Tests that HostBlock owns aligned host memory and hands it over on a move.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <utility>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/host_block.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_HostBlock, ItsMemoryStartsOnTheRequestedAlignment)
        {
            const HostBlock block(300, 256);

            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(block.data()) % 256, std::uintptr_t{0});
        }

        TEST(Small_HostBlock, ItReportsTheBytesRequested)
        {
            const HostBlock block(300, 128);

            EXPECT_EQ(block.size_bytes(), std::size_t{300});
        }

        TEST(Small_HostBlock, ADefaultBlockOwnsNothing)
        {
            const HostBlock block;

            EXPECT_EQ(block.data(), nullptr);
            EXPECT_EQ(block.size_bytes(), std::size_t{0});
        }

        TEST(Small_HostBlock, AMoveHandsTheMemoryOverAndLeavesTheSourceEmpty)
        {
            HostBlock source(128, 128);
            void* const memory = source.data();

            const HostBlock target(std::move(source));

            EXPECT_EQ(target.data(), memory);
            EXPECT_EQ(target.size_bytes(), std::size_t{128});
            EXPECT_EQ(source.data(), nullptr);
            EXPECT_EQ(source.size_bytes(), std::size_t{0});
        }

        TEST(Small_HostBlock, AMoveAssignmentTakesTheMemoryOfTheSource)
        {
            HostBlock source(128, 128);
            HostBlock target(64, 64);
            void* const memory = source.data();

            target = std::move(source);

            EXPECT_EQ(target.data(), memory);
            EXPECT_EQ(target.size_bytes(), std::size_t{128});
            EXPECT_EQ(source.data(), nullptr);
        }
    } // namespace Test
} // namespace SushiBLAS
