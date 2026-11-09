/**************************************************************************/
/* test_profile_session.cpp                                               */
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
 * @file test_profile_session.cpp
 * @brief Tests that a benchmark program records nothing unless it is asked to.
 * @author Mustafa Garip
 * @see tests/operations/benchmark_harness/README.md
 */

#include <optional>
#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/profile/profile_gate.hpp>

#if SB_ENABLE_PROFILER
#include <SushiBLAS/benchmark_harness/profile_session.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief A lookup that finds no variable. */
            std::optional<std::string> nothing_is_set(const char*)
            {
                return std::nullopt;
            }

            /** @brief A lookup that finds every variable empty. */
            std::optional<std::string> everything_is_empty(const char*)
            {
                return std::string();
            }
        } // namespace

        TEST(Medium_ProfileSession, WithoutTheFolderVariableNoSessionStarts)
        {
            Bench::ProfileSession* session = Bench::ProfileSession::start_from_environment(
                TestEnvironment::runtime(), "test", nothing_is_set);

            EXPECT_EQ(session, nullptr);
            EXPECT_EQ(Bench::ProfileSession::current(), nullptr);
        }

        TEST(Medium_ProfileSession, AnEmptyFolderVariableStartsNoSession)
        {
            Bench::ProfileSession* session = Bench::ProfileSession::start_from_environment(
                TestEnvironment::runtime(), "test", everything_is_empty);

            EXPECT_EQ(session, nullptr);
            EXPECT_EQ(Bench::ProfileSession::current(), nullptr);
        }

        TEST(Small_ProfileSession, MarkingAnIterationWithNoSessionDoesNothing)
        {
            EXPECT_FALSE(Bench::ProfileSession::mark(nullptr));
        }
    } // namespace Test
} // namespace SushiBLAS

#endif // SB_ENABLE_PROFILER
