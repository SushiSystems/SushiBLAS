/**************************************************************************/
/* test_recorded_graph.cpp                                                */
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
 * @file test_recorded_graph.cpp
 * @brief Tests that a RecordedGraph counts, runs and discards what its recorder recorded.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/recorded_graph.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Records one task that adds one to a counter into a graph. */
        static void record_increment(TaskRecorder& recorder, int* counter)
        {
            TaskDescription description;
            description.name = "increment";
            recorder.record_pair(description, OperandSet(), "test",
                                 [counter](const KernelPairDispatcher& dispatcher,
                                           sycl::queue& q, const std::vector<sycl::event>& deps)
                                 {
                                     return dispatcher.run(
                                         q, deps, [counter]() { *counter += 1; },
                                         [counter](sycl::queue& queue,
                                                   const std::vector<sycl::event>& events)
                                         {
                                             return queue.single_task(events, [=]() {
                                                 *counter += 1;
                                             });
                                         });
                                 });
        }

        TEST(Medium_RecordedGraph, StartsEmptyAndRunsNothing)
        {
            RecordedGraph graph(TestEnvironment::runtime());

            EXPECT_TRUE(graph.is_empty());
            EXPECT_EQ(graph.recorded_count(), std::size_t{0});
            EXPECT_TRUE(graph.run().has_value());
        }

        TEST(Medium_RecordedGraph, RunsWhatWasRecordedAndIsEmptyAfterwards)
        {
            RecordedGraph graph(TestEnvironment::runtime());
            ScratchPool pool(TestEnvironment::runtime());
            TaskRecorder recorder(graph.recorder(), pool);
            // The two tasks share no operand, so each gets a counter of its own.
            int* counters = sycl::malloc_shared<int>(2, TestEnvironment::queue());
            counters[0] = 0;
            counters[1] = 0;

            record_increment(recorder, counters);
            record_increment(recorder, counters + 1);
            EXPECT_EQ(graph.recorded_count(), std::size_t{2});

            ASSERT_TRUE(graph.run().has_value());
            ASSERT_TRUE(TestEnvironment::runtime().wait().has_value());

            EXPECT_EQ(counters[0], 1);
            EXPECT_EQ(counters[1], 1);
            EXPECT_TRUE(graph.is_empty());
            sycl::free(counters, TestEnvironment::queue());
        }

        TEST(Medium_RecordedGraph, DiscardDropsTheRecordedTasksAndCountsThem)
        {
            RecordedGraph graph(TestEnvironment::runtime());
            ScratchPool pool(TestEnvironment::runtime());
            TaskRecorder recorder(graph.recorder(), pool);
            int counter = 0;

            record_increment(recorder, &counter);

            EXPECT_EQ(graph.discard(), std::size_t{1});
            EXPECT_TRUE(graph.is_empty());
            EXPECT_EQ(counter, 0);
        }
    } // namespace Test
} // namespace SushiBLAS
