/**************************************************************************/
/* test_scratch_region.cpp                                                */
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


/**
 * @file test_scratch_region.cpp
 * @brief Tests ScratchRegion and name_scratch() over the host recorder, without a runtime.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include <SushiBLAS/graph_primitives/host_primitive_recorder.hpp>
#include <SushiBLAS/graph_primitives/scratch_region.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the words the rows allocate.
            constexpr std::size_t K_WORDS = 8;
        } // namespace

        /** @brief Tests that a named block is one handle and its windows cut bytes from it. */
        TEST(Small_GraphPrimitivesScratchRegion, WindowsCutBytesFromTheHandleOfTheBlock)
        {
            GraphPrimitives::HostPrimitiveRecorder recorder;
            const std::uint32_t* block =
                *recorder.scratch<std::uint32_t>(K_WORDS, SushiRuntime::DeviceIndex{},
                                                 SushiRuntime::Residency::Host);

            auto named = GraphPrimitives::Detail::name_scratch(recorder, block, K_WORDS,
                                                               SushiRuntime::DeviceIndex{});
            ASSERT_TRUE(named);
            const SushiRuntime::BufferRegion window = named->window(2, 3);

            EXPECT_TRUE(named->whole().valid());
            EXPECT_EQ(window.base, named->whole().base);
            EXPECT_EQ(window.offset, 2 * sizeof(std::uint32_t));
            EXPECT_EQ(window.length, 3 * sizeof(std::uint32_t));
        }

        /** @brief Tests that disjoint windows of one block do not overlap and equal ones do. */
        TEST(Small_GraphPrimitivesScratchRegion, DisjointWindowsDoNotOverlap)
        {
            GraphPrimitives::HostPrimitiveRecorder recorder;
            const std::uint32_t* block =
                *recorder.scratch<std::uint32_t>(K_WORDS, SushiRuntime::DeviceIndex{},
                                                 SushiRuntime::Residency::Host);
            const auto named = *GraphPrimitives::Detail::name_scratch(
                recorder, block, K_WORDS, SushiRuntime::DeviceIndex{});

            EXPECT_FALSE(named.window(0, 4).overlaps(named.window(4, 4)));
            EXPECT_TRUE(named.window(0, 5).overlaps(named.window(4, 4)));
        }

        /** @brief Tests that two blocks get two handles. */
        TEST(Small_GraphPrimitivesScratchRegion, TwoBlocksHaveTwoHandles)
        {
            GraphPrimitives::HostPrimitiveRecorder recorder;
            const std::uint32_t* first =
                *recorder.scratch<std::uint32_t>(K_WORDS, SushiRuntime::DeviceIndex{},
                                                 SushiRuntime::Residency::Host);
            const std::uint32_t* second =
                *recorder.scratch<std::uint32_t>(K_WORDS, SushiRuntime::DeviceIndex{},
                                                 SushiRuntime::Residency::Host);

            const auto a = *GraphPrimitives::Detail::name_scratch(recorder, first, K_WORDS,
                                                                  SushiRuntime::DeviceIndex{});
            const auto b = *GraphPrimitives::Detail::name_scratch(recorder, second, K_WORDS,
                                                                  SushiRuntime::DeviceIndex{});

            EXPECT_NE(a.whole().base, b.whole().base);
        }

        /** @brief Tests that a null block or a zero count names nothing. */
        TEST(Small_GraphPrimitivesScratchRegion, ANullBlockOrAZeroCountNamesNothing)
        {
            GraphPrimitives::HostPrimitiveRecorder recorder;
            const std::uint32_t* block =
                *recorder.scratch<std::uint32_t>(K_WORDS, SushiRuntime::DeviceIndex{},
                                                 SushiRuntime::Residency::Host);

            const auto none = *GraphPrimitives::Detail::name_scratch<
                GraphPrimitives::HostPrimitiveRecorder, std::uint32_t>(
                recorder, nullptr, K_WORDS, SushiRuntime::DeviceIndex{});
            const auto empty = *GraphPrimitives::Detail::name_scratch(
                recorder, block, std::size_t{0}, SushiRuntime::DeviceIndex{});

            EXPECT_FALSE(none.whole().valid());
            EXPECT_FALSE(none.window(0, 1).valid());
            EXPECT_FALSE(empty.whole().valid());
        }

        /** @brief Tests that an address no scratch() call returned is refused. */
        TEST(Small_GraphPrimitivesScratchRegion, AnUnknownBlockIsRefused)
        {
            GraphPrimitives::HostPrimitiveRecorder recorder;
            const std::uint32_t stray[2] = {0, 0};

            const auto named = GraphPrimitives::Detail::name_scratch(
                recorder, stray, std::size_t{2}, SushiRuntime::DeviceIndex{});

            EXPECT_FALSE(named);
        }
    } // namespace Test
} // namespace SushiBLAS
