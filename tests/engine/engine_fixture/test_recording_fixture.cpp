/**************************************************************************/
/* test_recording_fixture.cpp                                             */
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
 * @file test_recording_fixture.cpp
 * @brief Tests that RecordingFixture forwards to its RecordingGraph and asserts on a run.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/engine_fixture/recording_fixture.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Records a task that does nothing and reads or writes nothing. */
        static void record_nothing(RecordingFixture& fixture)
        {
            TaskDescription description;
            description.name = "nothing";
            fixture.recorder().record_pair(description, OperandSet(), "test",
                                           [](const KernelPairDispatcher& dispatcher,
                                              sycl::queue& q,
                                              const std::vector<sycl::event>& deps)
                                           {
                                               return dispatcher.run(
                                                   q, deps,
                                                   []()
                                                   {

                                                   },
                                                   [](sycl::queue&,
                                                      const std::vector<sycl::event>& events)
                                                   {
                                                       sycl::event::wait(events);
                                                       return sycl::event();
                                                   });
                                           });
        }

        /** @brief Builds fixtures over the engine of the test fixture. */
        class Medium_RecordingFixture : public SushiBLASTest
        {

        };

        TEST_F(Medium_RecordingFixture, RunSucceedsAcceptsAnEmptyFixture)
        {
            RecordingFixture fixture(*engine);

            EXPECT_TRUE(fixture.run_succeeds());
        }

        TEST_F(Medium_RecordingFixture, RunSucceedsRunsTheRecordedTaskAndEmptiesTheFixture)
        {
            RecordingFixture fixture(*engine);
            record_nothing(fixture);
            EXPECT_EQ(fixture.recorded_count(), std::size_t{1});

            EXPECT_TRUE(fixture.run_succeeds());
            EXPECT_TRUE(fixture.is_empty());
        }

        TEST_F(Medium_RecordingFixture, DiscardDropsTheTasksAndCountsThem)
        {
            RecordingFixture fixture(*engine);
            record_nothing(fixture);
            record_nothing(fixture);

            EXPECT_EQ(fixture.discard(), std::size_t{2});
            EXPECT_TRUE(fixture.is_empty());
        }

        TEST_F(Medium_RecordingFixture, TheRecorderBorrowsTheModeOfTheEngine)
        {
            engine->set_mode(ExecutionMode::ForceReference);
            RecordingFixture fixture(*engine);

            EXPECT_EQ(fixture.recorder().mode(), ExecutionMode::ForceReference);
        }
    } // namespace Test
} // namespace SushiBLAS
