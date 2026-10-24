/**************************************************************************/
/* test_primitive_options.cpp                                             */
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
 * @file test_primitive_options.cpp
 * @brief Tests the check that refuses options which name no node or hold no class of the runtime.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <optional>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include <SushiRuntime/core/determinism.hpp>
#include <SushiRuntime/core/error.hpp>

#include <SushiBLAS/graph_primitives/primitive_options.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using GraphPrimitives::PrimitiveOptions;
            using GraphPrimitives::Detail::require_options;
        } // namespace

        /** @brief Tests that a call with no options passes the check. */
        TEST(Small_GraphPrimitivesPrimitiveOptions, NoOptionsPass)
        {
            EXPECT_TRUE(require_options("reduce", std::nullopt).has_value());
        }

        /** @brief Tests that a name with each class of the runtime passes the check. */
        TEST(Small_GraphPrimitivesPrimitiveOptions, ANamedClassPasses)
        {
            for (const auto determinism :
                 {SushiRuntime::Determinism::Unstated, SushiRuntime::Determinism::Bitwise,
                  SushiRuntime::Determinism::Tolerant, SushiRuntime::Determinism::Cosmetic})
                EXPECT_TRUE(require_options("reduce", PrimitiveOptions{"sum", determinism})
                                .has_value());
        }

        /** @brief Tests that a null or empty name is refused, naming the primitive and 'options'. */
        TEST(Small_GraphPrimitivesPrimitiveOptions, ANullOrEmptyNameIsRefused)
        {
            for (const std::string_view name : {std::string_view{}, std::string_view{""}})
            {
                const auto refused = require_options("prefix_scan", PrimitiveOptions{name});
                ASSERT_FALSE(refused.has_value());
                EXPECT_EQ(refused.error().code, SushiRuntime::ErrorCode::InvalidGraph);
                EXPECT_EQ(refused.error().message.rfind("prefix_scan:", 0), 0u)
                    << refused.error().message;
                EXPECT_NE(refused.error().message.find("'options'"), std::string::npos)
                    << refused.error().message;
            }
        }

        /** @brief Tests that a class outside the runtime's enumeration is refused. */
        TEST(Small_GraphPrimitivesPrimitiveOptions, AClassOutsideTheEnumerationIsRefused)
        {
            const auto refused = require_options(
                "reduce", PrimitiveOptions{"sum", static_cast<SushiRuntime::Determinism>(250)});
            ASSERT_FALSE(refused.has_value());
            EXPECT_EQ(refused.error().code, SushiRuntime::ErrorCode::InvalidGraph);
            EXPECT_NE(refused.error().message.find("'options'"), std::string::npos)
                << refused.error().message;
        }
    } // namespace Test
} // namespace SushiBLAS
