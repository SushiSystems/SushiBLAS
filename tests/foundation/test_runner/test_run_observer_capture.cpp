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
 * @file test_run_observer_capture.cpp
 * @brief Tests that RunObserverCapture keeps one record per dispatch of each run it observed.
 * @author Mustafa Garip
 * @see tests/foundation/test_runner/README.md
 */

#include <cstddef>
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

        TEST(Medium_RunObserverCapture, KeepsOneRecordPerDispatch)
        {
            SushiRuntime::Runtime& runtime = TestEnvironment::runtime();
            RunObserverCapture capture(runtime);

            run_two_nodes(runtime);

            EXPECT_EQ(capture.run_count(), std::size_t{1});
            EXPECT_EQ(capture.records().size(), std::size_t{2});
        }

        TEST(Medium_RunObserverCapture, RecordsArriveInTheRunsOrderAndClearForgetsThem)
        {
            SushiRuntime::Runtime& runtime = TestEnvironment::runtime();
            RunObserverCapture capture(runtime);

            run_two_nodes(runtime);
            run_two_nodes(runtime);
            EXPECT_EQ(capture.run_count(), std::size_t{2});
            EXPECT_EQ(capture.records().size(), std::size_t{4});

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
            SUCCEED();
        }
    } // namespace Test
} // namespace SushiBLAS
