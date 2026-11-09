/**************************************************************************/
/* test_operation_options.cpp                                             */
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
 * @file test_operation_options.cpp
 * @brief Tests that OperationOptions names a node and states no determinism class unless told to,
 * and the record-time check every operation family shares.
 * @author Mustafa Garip
 */

#include <string>
#include <cstdint>
#include <string_view>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/operation_options.hpp>
#include <SushiRuntime/core/determinism.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_OperationOptions, ANameAloneStatesNoDeterminismClass)
        {
            const OperationOptions options{"pgs_project"};

            EXPECT_EQ(options.name, "pgs_project");
            EXPECT_EQ(options.determinism, SushiRuntime::Determinism::Unstated);
        }

        namespace
        {
            /** @brief Names one class of the runtime. */
            struct ClassRow
            {
                const char* name;                 /**< The row name a failure shows. */
                SushiRuntime::Determinism held;   /**< The class the options state. */
            };

            /** @brief Returns one row per class of the runtime. */
            std::vector<ClassRow> class_rows()
            {
                return {{"Unstated", SushiRuntime::Determinism::Unstated},
                        {"Bitwise", SushiRuntime::Determinism::Bitwise},
                        {"Tolerant", SushiRuntime::Determinism::Tolerant},
                        {"Cosmetic", SushiRuntime::Determinism::Cosmetic}};
            }

            /** @brief Returns whether the shared check refuses the options. */
            bool is_refused(const OperationOptions& options)
            {
                try
                {
                    Dispatch::require_operation_options(options, "op");
                }
                catch (const std::invalid_argument&)
                {
                    return true;
                }

                return false;
            }

            /** @brief Returns the text of the refusal the shared check gives an empty name, or "". */
            std::string refusal_of_an_empty_name()
            {
                try
                {
                    Dispatch::require_operation_options(OperationOptions{""}, "family.verb");
                }
                catch (const std::invalid_argument& error)
                {
                    return error.what();
                }

                return std::string();
            }
        } // namespace

        /** @brief Runs one row of class_rows() through the shared check. */
        class Small_OperationOptionsClass : public ::testing::TestWithParam<ClassRow>
        {

        };

        TEST_P(Small_OperationOptionsClass, TheSharedCheckAcceptsTheClass)
        {
            EXPECT_FALSE(is_refused(OperationOptions{"node", GetParam().held}));
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_OperationOptionsClass, ::testing::ValuesIn(class_rows()),
            [](const ::testing::TestParamInfo<ClassRow>& info)
            { return std::string(info.param.name); });

        TEST(Small_OperationOptions, TheSharedCheckRefusesAnEmptyOrNullName)
        {
            EXPECT_THROW(Dispatch::require_operation_options(OperationOptions{""}, "op"),
                         std::invalid_argument);
            EXPECT_THROW(Dispatch::require_operation_options(OperationOptions{}, "op"),
                         std::invalid_argument);
            EXPECT_THROW(Dispatch::require_operation_options(
                             OperationOptions{std::string_view(nullptr, 0)}, "op"),
                         std::invalid_argument);
        }

        TEST(Small_OperationOptions, TheSharedCheckRefusesAClassOutsideTheEnum)
        {
            const OperationOptions options{
                "node", static_cast<SushiRuntime::Determinism>(
                            static_cast<std::uint8_t>(SushiRuntime::Determinism::Cosmetic) + 1)};

            EXPECT_THROW(Dispatch::require_operation_options(options, "op"),
                         std::invalid_argument);
        }

        TEST(Small_OperationOptions, TheMessageNamesTheOperationAndOptions)
        {
            const std::string text = refusal_of_an_empty_name();

            EXPECT_NE(text.find("family.verb"), std::string::npos) << text;
            EXPECT_NE(text.find("'options'"), std::string::npos) << text;
        }
    } // namespace Test
} // namespace SushiBLAS
