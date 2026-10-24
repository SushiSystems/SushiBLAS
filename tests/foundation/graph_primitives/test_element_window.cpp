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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
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

        /** @brief Tests that windows ending at or before the buffer's end fit it. */
        TEST(Small_GraphPrimitivesElementWindow, AcceptsWindowsInsideTheBuffer)
        {
            EXPECT_TRUE(require_window("p", "input", 8, 0, 8).has_value());
            EXPECT_TRUE(require_window("p", "input", 8, 3, 5).has_value());
            EXPECT_TRUE(require_window("p", "input", 8, 8, 0).has_value());
            EXPECT_TRUE(require_window("p", "input", 0, 0, 0).has_value());
        }

        /** @brief Tests that a window past the end is refused, even where offset + count wraps. */
        TEST(Small_GraphPrimitivesElementWindow, RefusesWindowsPastTheBuffer)
        {
            const std::size_t huge = std::numeric_limits<std::size_t>::max();
            for (const auto& window : {std::pair<std::size_t, std::size_t>{9, 0},
                                       std::pair<std::size_t, std::size_t>{4, 5},
                                       std::pair<std::size_t, std::size_t>{1, huge}})
            {
                const SushiRuntime::Result<void> refused =
                    require_window("reduce", "output", 8, window.first, window.second);
                ASSERT_FALSE(refused.has_value()) << window.first << " + " << window.second;
                EXPECT_EQ(refused.error().code, SushiRuntime::ErrorCode::InvalidGraph);
                EXPECT_EQ(refused.error().message.rfind("reduce: the output window", 0), 0u)
                    << refused.error().message;
            }
        }

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
            const SushiRuntime::Result<void> refused = require_window("reduce", "output", 8, 1, huge);
            ASSERT_FALSE(refused.has_value());
            EXPECT_NE(refused.error().message.find("window [1, past the largest size)"),
                      std::string::npos)
                << refused.error().message;
        }

        /** @brief Tests that a span narrows its buffer's key to its bytes and keeps the watch. */
        TEST(Medium_GraphPrimitivesElementWindow, AccessOfNarrowsSpansAndKeepsRegions)
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
            EXPECT_EQ(writes.lives.size(), SushiRuntime::Writes(buffer).lives.size())
                << "the span's buffer must be watched as the buffer itself is";
        }
    } // namespace Test
} // namespace SushiBLAS
