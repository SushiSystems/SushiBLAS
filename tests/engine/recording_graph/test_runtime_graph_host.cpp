/**************************************************************************/
/* test_runtime_graph_host.cpp                                            */
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
 * @file test_runtime_graph_host.cpp
 * @brief Tests that RuntimeGraphHost records on a runtime graph, orders its plan, runs it and releases the hold of its late-count storages.
 * @author Mustafa Garip
 * @see tests/engine/recording_graph/README.md
 */

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/recording_graph/recording_graph.hpp>
#include <SushiBLAS/recording_graph/runtime_graph_host.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include "recording_graph_tasks.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Builds a runtime and an engine over it for each test. */
        class Medium_RuntimeGraphHost : public ::testing::Test
        {
            protected:
                /// @brief The runtime the engine and the graphs run on.
                SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();

                /// @brief The engine whose scratch pool the graphs borrow.
                Engine engine{runtime};
        };

        TEST_F(Medium_RuntimeGraphHost, TheRuntimeConstructorRecordsUnderTheModeItIsGiven)
        {
            RecordingGraph graph(runtime, engine.scratch_pool(), ExecutionMode::ForceSYCL);
            RecordingGraph default_graph(runtime, engine.scratch_pool());

            EXPECT_EQ(graph.recorder().mode(), ExecutionMode::ForceSYCL);
            EXPECT_EQ(default_graph.recorder().mode(), K_DEFAULT_RECORDING_MODE);
        }

        TEST_F(Medium_RuntimeGraphHost, ARunEmptiesTheRuntimeGraph)
        {
            RecordingGraph graph(runtime, engine.scratch_pool(), ExecutionMode::ForceSYCL);
            record_nothing(graph);
            ASSERT_EQ(graph.recorded_count(), std::size_t{1});

            EXPECT_TRUE(graph.run().has_value());
            EXPECT_EQ(graph.recorded_count(), std::size_t{0});
        }

        TEST_F(Medium_RuntimeGraphHost, ThePlanOrdersOneEntryPerRecordedTask)
        {
            RecordingGraph graph(runtime, engine.scratch_pool(), ExecutionMode::ForceSYCL);
            record_nothing(graph);
            record_nothing(graph);

            const auto plan = graph.plan();

            ASSERT_TRUE(plan.has_value());
            EXPECT_EQ(plan->nodes.size(), std::size_t{2});
            EXPECT_EQ(graph.recorded_count(), std::size_t{2});
        }

        /** @brief Runs the RuntimeGraphHost tests that guard the release of late-count storage. */
        class Medium_Regression_RuntimeGraphHost : public Medium_RuntimeGraphHost
        {

        };

        /**
         * @brief Tests that after a run the hold keeps no late-count token, so dropping every
         *        tensor frees the storage.
         */
        TEST_F(Medium_Regression_RuntimeGraphHost, ARunLetsGoOfTheStoragesOfItsLateCountTasks)
        {
            RecordingGraph graph(runtime, engine.scratch_pool(), ExecutionMode::ForceSYCL);
            Tensor out = engine.create_tensor({4}, Core::DataType::Float32);
            Tensor count = engine.create_tensor({1}, Core::DataType::Int32,
                                                SushiRuntime::Residency::Host);
            count.data_as<std::int32_t>()[0] = 2;

            record_late_task(graph, out, count);
            ASSERT_EQ(graph.late_counts()->held_count(), std::size_t{1});
            EXPECT_GT(out.storage->ref_count(), 1);

            ASSERT_TRUE(graph.run().has_value());
            EXPECT_EQ(graph.late_counts()->held_count(), std::size_t{0});
            EXPECT_EQ(out.storage->ref_count(), 1);
            EXPECT_EQ(count.storage->ref_count(), 1);
        }

        /** @brief Tests that a discard lets go of the storages of its late-count tasks. */
        TEST_F(Medium_Regression_RuntimeGraphHost, ADiscardLetsGoOfTheStoragesOfItsLateCountTasks)
        {
            RecordingGraph graph(runtime, engine.scratch_pool(), ExecutionMode::ForceSYCL);
            Tensor out = engine.create_tensor({4}, Core::DataType::Float32);
            Tensor count = engine.create_tensor({1}, Core::DataType::Int32,
                                                SushiRuntime::Residency::Host);
            count.data_as<std::int32_t>()[0] = 2;

            record_late_task(graph, out, count);
            EXPECT_EQ(graph.discard(), std::size_t{1});
            EXPECT_EQ(graph.late_counts()->held_count(), std::size_t{0});
            EXPECT_EQ(out.storage->ref_count(), 1);
            EXPECT_EQ(count.storage->ref_count(), 1);
        }
    } // namespace Test
} // namespace SushiBLAS
