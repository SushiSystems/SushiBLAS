/**************************************************************************/
/* test_host_buffer.cpp                                                   */
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
 * @file test_host_buffer.cpp
 * @brief Tests the size, zeroed memory, region and indexing of HostBuffer, without a runtime.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <cstddef>
#include <cstdint>
#include <utility>

#include <gtest/gtest.h>

#include <SushiRuntime/region/buffer_region.hpp>

#include <SushiBLAS/graph_primitives/host_buffer.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the elements the rows allocate.
            constexpr std::size_t K_COUNT = 6;
        } // namespace

        /** @brief Tests that a buffer reports the count it was asked for. */
        TEST(Small_GraphPrimitivesHostBuffer, ReportsItsSize)
        {
            const GraphPrimitives::HostBuffer<std::uint32_t> buffer(K_COUNT);

            EXPECT_EQ(buffer.size(), K_COUNT);
        }

        /** @brief Tests that every element starts at zero. */
        TEST(Small_GraphPrimitivesHostBuffer, StartsZeroed)
        {
            const GraphPrimitives::HostBuffer<std::uint32_t> buffer(K_COUNT);

            for (std::size_t i = 0; i < K_COUNT; ++i)
                EXPECT_EQ(buffer[i], 0u) << "element " << i;
        }

        /** @brief Tests that a write through data() reads back through the index. */
        TEST(Small_GraphPrimitivesHostBuffer, DataAndIndexNameTheSameElements)
        {
            const GraphPrimitives::HostBuffer<std::uint32_t> buffer(K_COUNT);

            buffer.data()[2] = 7u;

            EXPECT_EQ(buffer[2], 7u);
        }

        /** @brief Tests that an empty buffer still has an address a region can key on. */
        TEST(Small_GraphPrimitivesHostBuffer, AnEmptyBufferHasAValidRegion)
        {
            const GraphPrimitives::HostBuffer<std::uint32_t> buffer(0);

            EXPECT_EQ(buffer.size(), std::size_t{0});
            EXPECT_TRUE(buffer.region().valid());
        }

        /** @brief Tests that the region keys on the first element and spans the whole buffer. */
        TEST(Small_GraphPrimitivesHostBuffer, TheRegionKeysOnTheDataAndSpansTheBuffer)
        {
            const GraphPrimitives::HostBuffer<std::uint32_t> buffer(K_COUNT);

            const SushiRuntime::BufferRegion region = buffer.region();

            EXPECT_EQ(region.base, static_cast<void*>(buffer.data()));
            EXPECT_EQ(region.offset, std::size_t{0});
            EXPECT_EQ(region.length, SushiRuntime::BufferRegion::K_WHOLE);
        }

        /** @brief Tests that moving a buffer keeps the address of its elements. */
        TEST(Small_GraphPrimitivesHostBuffer, MovingKeepsTheAddressOfTheElements)
        {
            GraphPrimitives::HostBuffer<std::uint32_t> buffer(K_COUNT);
            const std::uint32_t* before = buffer.data();

            const GraphPrimitives::HostBuffer<std::uint32_t> moved(std::move(buffer));

            EXPECT_EQ(moved.data(), before);
        }
    } // namespace Test
} // namespace SushiBLAS
