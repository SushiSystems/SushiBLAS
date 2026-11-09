/**************************************************************************/
/* test_environment_value.cpp                                               */
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
 * @file test_environment_value.cpp
 * @brief Tests that environment_value reads a variable, and returns empty for an unset or empty one.
 * @author Mustafa Garip
 * @see tests/operations/benchmark_harness/README.md
 */

#include <map>
#include <optional>
#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/benchmark_harness/environment_value.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            constexpr const char* K_VARIABLE = "SB_TEST_ENVIRONMENT_VALUE";

            /** @brief A lookup over a map the test owns; a name the map lacks is unset. */
            Bench::EnvironmentLookup lookup_over(const std::map<std::string, std::string>& variables)
            {
                return [&variables](const char* name) -> std::optional<std::string>
                {
                    const auto found = variables.find(name);
                    if (found == variables.end())
                        return std::nullopt;
                    return found->second;
                };
            }
        } // namespace

        TEST(Small_EnvironmentValue, ReturnsTheValueOfASetVariable)
        {
            const std::map<std::string, std::string> variables{{K_VARIABLE, "fixed-value"}};

            EXPECT_EQ(Bench::environment_value(K_VARIABLE, lookup_over(variables)), "fixed-value");
        }

        TEST(Small_EnvironmentValue, ReturnsEmptyForAnUnsetVariable)
        {
            const std::map<std::string, std::string> variables;

            EXPECT_TRUE(Bench::environment_value(K_VARIABLE, lookup_over(variables)).empty());
        }

        TEST(Small_EnvironmentValue, ReturnsEmptyForAnEmptyVariable)
        {
            const std::map<std::string, std::string> variables{{K_VARIABLE, ""}};

            EXPECT_TRUE(Bench::environment_value(K_VARIABLE, lookup_over(variables)).empty());
        }

        TEST(Small_EnvironmentValue, KeepsASpaceInsideTheValue)
        {
            const std::map<std::string, std::string> variables{{K_VARIABLE, "two words"}};

            EXPECT_EQ(Bench::environment_value(K_VARIABLE, lookup_over(variables)), "two words");
        }

        TEST(Small_EnvironmentValue, ReflectsTheLatestValueOnEveryCall)
        {
            std::map<std::string, std::string> variables{{K_VARIABLE, "first"}};
            const Bench::EnvironmentLookup lookup = lookup_over(variables);

            const std::string first = Bench::environment_value(K_VARIABLE, lookup);
            variables[K_VARIABLE] = "second";

            EXPECT_EQ(first, "first");
            EXPECT_EQ(Bench::environment_value(K_VARIABLE, lookup), "second");
        }
    } // namespace Test
} // namespace SushiBLAS
