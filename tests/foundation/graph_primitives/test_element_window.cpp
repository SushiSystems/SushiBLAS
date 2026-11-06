/**************************************************************************/
/* test_element_window.cpp                                                */
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
 * @file test_element_window.cpp
 * @brief Tests the window check and the narrowed access of element_window.hpp.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <SushiRuntime/api/runtime/runtime.hpp>
#include <SushiRuntime/api/vocabulary/dependencies.hpp>
#include <SushiRuntime/core/error.hpp>
#include <SushiRuntime/region/buffer_region.hpp>

#include <SushiBLAS/graph_primitives/element_window.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        using GraphPrimitives::Detail::require_window;

        namespace
        {
            /// @brief Names a buffer size and a window over it.
            struct WindowRow
            {
                /// @brief Holds the buffer size in elements.
                std::size_t size;
                /// @brief Holds the first element of the window.
                std::size_t offset;
                /// @brief Holds the element count of the window.
                std::size_t count;
            };
        } // namespace

        /// @brief Runs one window against a buffer of eight elements.
        class Small_GraphPrimitivesElementWindowAccepted : public ::testing::TestWithParam<WindowRow>
        {
        };

        /** @brief Tests that a window ending at or before the buffer's end fits it. */
        TEST_P(Small_GraphPrimitivesElementWindowAccepted, AcceptsWindowsInsideTheBuffer)
        {
            const WindowRow& row = GetParam();

            EXPECT_TRUE(require_window("p", "input", row.size, row.offset, row.count).has_value());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesElementWindowAccepted,
                                 ::testing::Values(WindowRow{8, 0, 8}, WindowRow{8, 3, 5},
                                                   WindowRow{8, 8, 0}, WindowRow{0, 0, 0}));

        /// @brief Runs one window past the end of a buffer of eight elements.
        class Small_GraphPrimitivesElementWindowRefused : public ::testing::TestWithParam<WindowRow>
        {
        };

        /** @brief Tests that a window past the end is refused, even where offset + count wraps. */
        TEST_P(Small_GraphPrimitivesElementWindowRefused, RefusesWindowsPastTheBuffer)
        {
            const WindowRow& row = GetParam();

            const SushiRuntime::Result<void> refused =
                require_window("reduce", "output", row.size, row.offset, row.count);

            ASSERT_FALSE(refused.has_value());
            EXPECT_EQ(refused.error().code, SushiRuntime::ErrorCode::InvalidGraph);
            EXPECT_EQ(refused.error().message.rfind("reduce: the output window", 0), 0u)
                << refused.error().message;
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_GraphPrimitivesElementWindowRefused,
            ::testing::Values(WindowRow{8, 9, 0}, WindowRow{8, 4, 5},
                              WindowRow{8, 1, std::numeric_limits<std::size_t>::max()}));

        /** @brief Tests that a refused window prints its computed end, not the sum unworked. */
        TEST(Small_Regression_GraphPrimitivesElementWindow, MessagePrintsTheComputedEnd)
        {
            const SushiRuntime::Result<void> refused = require_window("reduce", "output", 7, 3, 5);
            ASSERT_FALSE(refused.has_value());
            EXPECT_NE(refused.error().message.find("window [3, 8)"), std::string::npos)
                << refused.error().message;
        }

        /** @brief Tests that a window whose end wraps says so instead of printing a wrapped end. */
        TEST(Small_Regression_GraphPrimitivesElementWindow, MessageNamesAnEndThatWraps)
        {
            const std::size_t huge = std::numeric_limits<std::size_t>::max();
            const SushiRuntime::Result<void> refused =
                require_window("reduce", "output", 8, 1, huge);
            ASSERT_FALSE(refused.has_value());
            EXPECT_NE(refused.error().message.find("window [1, past the largest size)"),
                      std::string::npos)
                << refused.error().message;
        }

        /** @brief Tests that a span narrows its buffer's key to its bytes and keeps the other keys. */
        TEST(Medium_GraphPrimitivesElementWindow, AccessOfNarrowsASpanToItsBytes)
        {
            auto runtime = SushiRuntime::Runtime::create().value();
            auto buffer = runtime.buffer<std::uint32_t>(16).value();
            const SushiRuntime::BufferRegion whole = SushiRuntime::Reads(buffer).keys.front();
            const SushiRuntime::BufferRegion other{&buffer, 4, 8};

            const SushiRuntime::Writes writes = GraphPrimitives::Detail::access_of<
                SushiRuntime::Writes>(GraphPrimitives::Detail::elements(buffer, 3, 5), other);

            ASSERT_EQ(writes.keys.size(), 2u);
            EXPECT_EQ(writes.keys[0].base, whole.base);
            EXPECT_EQ(writes.keys[0].offset, whole.offset + 3 * sizeof(std::uint32_t));
            EXPECT_EQ(writes.keys[0].length, 5 * sizeof(std::uint32_t));
            EXPECT_EQ(writes.keys[1].base, other.base);
            EXPECT_EQ(writes.keys[1].offset, other.offset);
            EXPECT_EQ(writes.keys[1].length, other.length);
        }

        /** @brief Tests that the buffer behind a span stays watched as the buffer itself is. */
        TEST(Medium_GraphPrimitivesElementWindow, AccessOfKeepsTheSpansBufferWatched)
        {
            auto runtime = SushiRuntime::Runtime::create().value();
            auto buffer = runtime.buffer<std::uint32_t>(16).value();

            const SushiRuntime::Writes writes = GraphPrimitives::Detail::access_of<
                SushiRuntime::Writes>(GraphPrimitives::Detail::elements(buffer, 3, 5));

            EXPECT_EQ(writes.lives.size(), SushiRuntime::Writes(buffer).lives.size());
        }
    } // namespace Test
} // namespace SushiBLAS
