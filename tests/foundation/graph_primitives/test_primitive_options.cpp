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
 * @file test_primitive_options.cpp
 * @brief Tests the check that refuses options which name no node or hold no class of the runtime.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <optional>
#include <string>
#include <string_view>
#include <vector>

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

            /// @brief Names a class of the runtime.
            struct ClassRow
            {
                /// @brief Names the row in the test report.
                const char* name;
                /// @brief Holds the class the options carry.
                SushiRuntime::Determinism determinism;
            };

            /// @brief Names an option name that names no node.
            struct NameRow
            {
                /// @brief Names the row in the test report.
                const char* label;
                /// @brief Holds the name the options carry.
                std::string_view name;
            };

            /** @brief Returns the four classes of the runtime. */
            std::vector<ClassRow> class_rows()
            {
                return {{"Unstated", SushiRuntime::Determinism::Unstated},
                        {"Bitwise", SushiRuntime::Determinism::Bitwise},
                        {"Tolerant", SushiRuntime::Determinism::Tolerant},
                        {"Cosmetic", SushiRuntime::Determinism::Cosmetic}};
            }

            /** @brief Returns the null and the empty name. */
            std::vector<NameRow> name_rows()
            {
                return {{"Null", std::string_view{}}, {"Empty", std::string_view{""}}};
            }
        } // namespace

        /** @brief Tests that a call with no options passes the check. */
        TEST(Small_GraphPrimitivesPrimitiveOptions, NoOptionsPass)
        {
            EXPECT_TRUE(require_options("reduce", std::nullopt).has_value());
        }

        /// @brief Runs one runtime class under a name.
        class Small_GraphPrimitivesPrimitiveOptionsClass : public ::testing::TestWithParam<ClassRow>
        {
        };

        /** @brief Tests that a name with a class of the runtime passes the check. */
        TEST_P(Small_GraphPrimitivesPrimitiveOptionsClass, ANamedClassPasses)
        {
            EXPECT_TRUE(
                require_options("reduce", PrimitiveOptions{"sum", GetParam().determinism})
                    .has_value());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesPrimitiveOptionsClass,
                                 ::testing::ValuesIn(class_rows()),
                                 [](const ::testing::TestParamInfo<ClassRow>& info)
                                 { return std::string(info.param.name); });

        /// @brief Runs one option set that names no node.
        class Small_GraphPrimitivesPrimitiveOptionsName : public ::testing::TestWithParam<NameRow>
        {
        };

        /** @brief Tests that a null or empty name is refused, naming the primitive and 'options'. */
        TEST_P(Small_GraphPrimitivesPrimitiveOptionsName, ANullOrEmptyNameIsRefused)
        {
            const auto refused = require_options("prefix_scan", PrimitiveOptions{GetParam().name});

            ASSERT_FALSE(refused.has_value());
            EXPECT_EQ(refused.error().code, SushiRuntime::ErrorCode::InvalidGraph);
            EXPECT_EQ(refused.error().message.rfind("prefix_scan:", 0), 0u)
                << refused.error().message;
            EXPECT_NE(refused.error().message.find("'options'"), std::string::npos)
                << refused.error().message;
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesPrimitiveOptionsName,
                                 ::testing::ValuesIn(name_rows()),
                                 [](const ::testing::TestParamInfo<NameRow>& info)
                                 { return std::string(info.param.label); });

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
