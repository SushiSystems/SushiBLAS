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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */

/**
 * @file test_environment_value.cpp
 * @brief Tests that environment_value reads a variable, and returns empty for an unset or empty one.
 * @author Mustafa Garip
 * @see tests/operations/benchmark_harness/README.md
 */

#include <cstdlib>
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

            /** @brief Sets or clears K_VARIABLE for one test and restores the earlier state. */
            class VariableScope
            {
            public:
                VariableScope()
                {
                    const std::string previous = Bench::environment_value(K_VARIABLE);
                    m_had_value = !previous.empty();
                    m_previous = previous;
                }

                ~VariableScope()
                {
                    if (m_had_value)
                        set(m_previous.c_str());
                    else
                        unset();
                }

                VariableScope(const VariableScope&) = delete;
                VariableScope& operator=(const VariableScope&) = delete;

                /** @brief Gives K_VARIABLE the value @p value. */
                static void set(const char* value)
                {
#if defined(_WIN32)
                    _putenv_s(K_VARIABLE, value);
#else
                    setenv(K_VARIABLE, value, 1);
#endif
                }

                /** @brief Removes K_VARIABLE from the environment. */
                static void unset()
                {
#if defined(_WIN32)
                    _putenv_s(K_VARIABLE, "");
#else
                    unsetenv(K_VARIABLE);
#endif
                }

            private:
                bool m_had_value = false;
                std::string m_previous;
            };
        } // namespace

        TEST(Small_EnvironmentValue, ReturnsTheValueOfASetVariable)
        {
            VariableScope scope;
            VariableScope::set("fixed-value");

            EXPECT_EQ(Bench::environment_value(K_VARIABLE), "fixed-value");
        }

        TEST(Small_EnvironmentValue, ReturnsEmptyForAnUnsetVariable)
        {
            VariableScope scope;
            VariableScope::unset();

            EXPECT_TRUE(Bench::environment_value(K_VARIABLE).empty());
        }

        TEST(Small_EnvironmentValue, ReturnsEmptyForAnEmptyVariable)
        {
            VariableScope scope;
            VariableScope::set("");

            EXPECT_TRUE(Bench::environment_value(K_VARIABLE).empty());
        }

        TEST(Small_EnvironmentValue, KeepsASpaceInsideTheValue)
        {
            VariableScope scope;
            VariableScope::set("two words");

            EXPECT_EQ(Bench::environment_value(K_VARIABLE), "two words");
        }

        TEST(Small_EnvironmentValue, ReflectsTheLatestValueOnEveryCall)
        {
            VariableScope scope;
            VariableScope::set("first");
            const std::string first = Bench::environment_value(K_VARIABLE);
            VariableScope::set("second");

            EXPECT_EQ(first, "first");
            EXPECT_EQ(Bench::environment_value(K_VARIABLE), "second");
        }
    } // namespace Test
} // namespace SushiBLAS
