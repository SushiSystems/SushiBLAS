/**************************************************************************/
/* test_recording_graph.cpp                                               */
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
 * @file test_recording_graph.cpp
 * @brief Tests that RecordingGraph records on its own graph, runs it and leaves it empty.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/recording_graph/recording_graph.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Records a task that does nothing and reads or writes nothing. */
        static void record_nothing(RecordingGraph& graph)
        {
            TaskDescription description;
            description.name = "nothing";
            graph.recorder().record_pair(description, OperandSet(), "test",
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

        /** @brief Builds a runtime and an engine over it for each test. */
        class Medium_RecordingGraph : public ::testing::Test
        {
            protected:
                /// @brief The runtime the engine and the graphs run on.
                SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();

                /// @brief The engine whose scratch pool and mode the graphs borrow.
                Engine engine{runtime};
        };

        TEST_F(Medium_RecordingGraph, AFreshGraphIsEmptyAndRunsNothing)
        {
            RecordingGraph graph(engine);

            EXPECT_TRUE(graph.is_empty());
            EXPECT_EQ(graph.recorded_count(), std::size_t{0});
            EXPECT_TRUE(graph.run().has_value());
        }

        TEST_F(Medium_RecordingGraph, RunningTheGraphEmptiesIt)
        {
            RecordingGraph graph(engine);
            record_nothing(graph);
            EXPECT_EQ(graph.recorded_count(), std::size_t{1});

            EXPECT_TRUE(graph.run().has_value());
            EXPECT_EQ(graph.recorded_count(), std::size_t{0});
        }

        TEST_F(Medium_RecordingGraph, DiscardDropsTheTasksAndCountsThem)
        {
            RecordingGraph graph(engine);
            record_nothing(graph);
            record_nothing(graph);

            EXPECT_EQ(graph.discard(), std::size_t{2});
            EXPECT_TRUE(graph.is_empty());
        }

        TEST_F(Medium_RecordingGraph, TheRecorderBorrowsTheModeOfTheEngine)
        {
            engine.set_mode(ExecutionMode::ForceReference);
            RecordingGraph graph(engine);

            EXPECT_EQ(graph.recorder().mode(), ExecutionMode::ForceReference);
        }

        TEST_F(Medium_RecordingGraph, TheRuntimeConstructorRecordsUnderTheModeItIsGiven)
        {
            RecordingGraph graph(runtime, engine.scratch_pool(), ExecutionMode::ForceSYCL);
            RecordingGraph default_graph(runtime, engine.scratch_pool());

            EXPECT_EQ(graph.recorder().mode(), ExecutionMode::ForceSYCL);
            EXPECT_EQ(default_graph.recorder().mode(), K_DEFAULT_RECORDING_MODE);
        }

        TEST_F(Medium_RecordingGraph, TheGraphRunsAgainAfterItWasEmptied)
        {
            RecordingGraph graph(engine);
            record_nothing(graph);
            ASSERT_TRUE(graph.run().has_value());

            record_nothing(graph);
            EXPECT_EQ(graph.recorded_count(), std::size_t{1});
            EXPECT_TRUE(graph.run().has_value());
        }

        TEST_F(Medium_RecordingGraph, ThePlanOrdersOneEntryPerRecordedTask)
        {
            RecordingGraph graph(engine);
            record_nothing(graph);
            record_nothing(graph);

            EXPECT_TRUE(graph.plan().has_value());
            EXPECT_EQ(graph.recorded_count(), std::size_t{2});
        }

        /** @brief Records one late-count task over out, whose width count holds, on graph. */
        static void record_late_task(RecordingGraph& graph, Tensor& out, const Tensor& count)
        {
            OperandSet operands;
            operands.writes(out);
            graph.recorder().record_late_pair(TaskDescription(), operands, "test", count,
                                              static_cast<std::size_t>(out.num_elements),
                                              [] {}, [](std::size_t) {});
        }

        /** @brief Runs the RecordingGraph tests that guard the release of late-count storage. */
        class Medium_Regression_RecordingGraph : public Medium_RecordingGraph
        {

        };

        /**
         * @brief Tests that after a run the graph holds no late-count storage, so dropping every
         *        tensor frees it.
         */
        TEST_F(Medium_Regression_RecordingGraph, ARunLetsGoOfTheStoragesOfItsLateCountTasks)
        {
            RecordingGraph graph(runtime, engine.scratch_pool(), ExecutionMode::ForceSYCL);
            Tensor out = engine.create_tensor({4}, Core::DataType::Float32);
            Tensor count = engine.create_tensor({1}, Core::DataType::Int32,
                                                SushiRuntime::Residency::Host);
            count.data_as<std::int32_t>()[0] = 2;

            record_late_task(graph, out, count);
            ASSERT_EQ(graph.late_counts().held_count(), std::size_t{1});
            EXPECT_GT(out.storage->ref_count(), 1);

            ASSERT_TRUE(graph.run().has_value());
            EXPECT_EQ(graph.late_counts().held_count(), std::size_t{0});
            EXPECT_EQ(out.storage->ref_count(), 1);
            EXPECT_EQ(count.storage->ref_count(), 1);
        }

        /** @brief Tests that a discard lets go of the storages of its late-count tasks. */
        TEST_F(Medium_Regression_RecordingGraph, ADiscardLetsGoOfTheStoragesOfItsLateCountTasks)
        {
            RecordingGraph graph(runtime, engine.scratch_pool(), ExecutionMode::ForceSYCL);
            Tensor out = engine.create_tensor({4}, Core::DataType::Float32);
            Tensor count = engine.create_tensor({1}, Core::DataType::Int32,
                                                SushiRuntime::Residency::Host);
            count.data_as<std::int32_t>()[0] = 2;

            record_late_task(graph, out, count);
            EXPECT_EQ(graph.discard(), std::size_t{1});
            EXPECT_EQ(graph.late_counts().held_count(), std::size_t{0});
            EXPECT_EQ(out.storage->ref_count(), 1);
            EXPECT_EQ(count.storage->ref_count(), 1);
        }
    } // namespace Test
} // namespace SushiBLAS
