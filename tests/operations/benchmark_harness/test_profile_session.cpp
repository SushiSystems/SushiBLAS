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
 * @file test_profile_session.cpp
 * @brief Tests that a benchmark program records nothing unless it is asked to.
 * @author Mustafa Garip
 * @see tests/operations/benchmark_harness/README.md
 */

#include <cstdlib>
#include <gtest/gtest.h>
#include <SushiBLAS/profile/profile_gate.hpp>

#if SB_ENABLE_PROFILER
#include <SushiBLAS/benchmark_harness/profile_session.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Medium_ProfileSession, WithoutTheFolderVariableNoSessionStarts)
        {
            if (!Bench::environment_value(Bench::K_PROFILE_FOLDER_VARIABLE).empty())
                GTEST_SKIP() << "SB_BENCH_PROFILE is set in this environment";

            Bench::ProfileSession* session =
                Bench::ProfileSession::start_from_environment(TestEnvironment::runtime(), "test");

            EXPECT_EQ(session, nullptr);
            EXPECT_EQ(Bench::ProfileSession::current(), nullptr);
        }

        TEST(Medium_ProfileSession, MarkingAnIterationWithNoSessionDoesNothing)
        {
            Bench::ProfileSession::mark_current_iteration();

            EXPECT_EQ(Bench::ProfileSession::current(), nullptr);
        }

        TEST(Medium_ProfileSession, TheIterationTimelineIsNamedAfterTheScenario)
        {
            EXPECT_STREQ(Bench::K_ITERATION_TIMELINE, "bench.iteration");
            EXPECT_STREQ(Bench::K_PROFILE_FOLDER_VARIABLE, "SB_BENCH_PROFILE");
        }
    } // namespace Test
} // namespace SushiBLAS

#endif // SB_ENABLE_PROFILER
