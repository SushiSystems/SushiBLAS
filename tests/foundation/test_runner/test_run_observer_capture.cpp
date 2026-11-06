/**************************************************************************/
/* test_run_observer_capture.cpp                                          */
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
 * @file test_run_observer_capture.cpp
 * @brief Tests that RunObserverCapture keeps one record per dispatch of each run it observed.
 * @author Mustafa Garip
 * @see tests/foundation/test_runner/README.md
 */

#include <cstddef>
#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/test_runner/run_observer_capture.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Records two one-element nodes on a fresh graph and runs it once.
            void run_two_nodes(SushiRuntime::Runtime& runtime)
            {
                auto graph = runtime.graph();
                for (int i = 0; i < 2; ++i)
                {
                    graph.add(std::size_t{1}, SushiRuntime::Reads(), SushiRuntime::Writes(),
                              [](std::size_t)
                              {

                              });
                }
                graph.run().value();
            }
        } // namespace

        /// @brief Holds the number of runs one case makes.
        using RunCount = std::size_t;

        /// @brief Names a row by its run count, which gtest takes as the test's suffix.
        static std::string run_count_name(const ::testing::TestParamInfo<RunCount>& info)
        {
            return std::to_string(info.param) + "Runs";
        }

        /// @brief Runs the two-node graph a number of times under one RunObserverCapture.
        class Medium_RunObserverCapture : public ::testing::TestWithParam<RunCount>
        {
        };

        TEST_P(Medium_RunObserverCapture, KeepsOneRecordPerDispatch)
        {
            SushiRuntime::Runtime& runtime = TestEnvironment::runtime();
            RunObserverCapture capture(runtime);

            for (RunCount run = 0; run < GetParam(); ++run)
            {
                run_two_nodes(runtime);
            }

            EXPECT_EQ(capture.run_count(), GetParam());
            EXPECT_EQ(capture.records().size(), GetParam() * 2);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_RunObserverCapture, ::testing::Values(RunCount{1}, RunCount{2}),
                                 run_count_name);

        TEST(Medium_RunObserverCaptureClear, ClearForgetsTheRunsAndTheirRecords)
        {
            SushiRuntime::Runtime& runtime = TestEnvironment::runtime();
            RunObserverCapture capture(runtime);
            run_two_nodes(runtime);
            run_two_nodes(runtime);

            capture.clear();

            EXPECT_EQ(capture.run_count(), std::size_t{0});
            EXPECT_TRUE(capture.records().empty());
        }

        TEST(Medium_RunObserverCapture, ARunAfterTheCaptureIsGoneIsNotKept)
        {
            SushiRuntime::Runtime& runtime = TestEnvironment::runtime();
            {
                RunObserverCapture capture(runtime);
                run_two_nodes(runtime);
            }

            run_two_nodes(runtime);

            RunObserverCapture fresh(runtime);

            EXPECT_EQ(fresh.run_count(), std::size_t{0});
        }
    } // namespace Test
} // namespace SushiBLAS
