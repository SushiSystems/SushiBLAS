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
 * @file test_recording_graph.cpp
 * @brief Tests that RecordingGraph records on its own host, runs it, leaves it empty and lets go of the storages of its late-count tasks.
 * @author Mustafa Garip
 * @see tests/engine/recording_graph/README.md
 */

#include <memory>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/recording_graph/host_graph_host.hpp>
#include <SushiBLAS/recording_graph/recording_graph.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include "recording_graph_tasks.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Builds a host engine and a graph over a host for each test. */
        class Small_RecordingGraph : public ::testing::Test
        {
            protected:
                /** @brief Returns a graph that records and runs on the host. */
                std::unique_ptr<RecordingGraph> make_graph()
                {
                    return std::make_unique<RecordingGraph>(std::make_unique<HostGraphHost>(),
                                                            engine.scratch_pool(),
                                                            ExecutionMode::ForceReference);
                }

                /// @brief The host engine whose scratch pool and tensors the graphs use.
                Engine engine{Engine::Host{}, Core::Layout::RowMajor};
        };

        TEST_F(Small_RecordingGraph, AFreshGraphIsEmptyAndRunsNothing)
        {
            const std::unique_ptr<RecordingGraph> graph = make_graph();

            EXPECT_TRUE(graph->is_empty());
            EXPECT_EQ(graph->recorded_count(), std::size_t{0});
            EXPECT_TRUE(graph->run().has_value());
        }

        TEST_F(Small_RecordingGraph, RunningTheGraphEmptiesIt)
        {
            const std::unique_ptr<RecordingGraph> graph = make_graph();
            record_nothing(*graph);
            EXPECT_EQ(graph->recorded_count(), std::size_t{1});

            EXPECT_TRUE(graph->run().has_value());
            EXPECT_EQ(graph->recorded_count(), std::size_t{0});
        }

        TEST_F(Small_RecordingGraph, DiscardDropsTheTasksAndCountsThem)
        {
            const std::unique_ptr<RecordingGraph> graph = make_graph();
            record_nothing(*graph);
            record_nothing(*graph);

            EXPECT_EQ(graph->discard(), std::size_t{2});
            EXPECT_TRUE(graph->is_empty());
        }

        TEST_F(Small_RecordingGraph, TheRecorderRecordsUnderTheModeItIsGiven)
        {
            const std::unique_ptr<RecordingGraph> graph = make_graph();

            EXPECT_EQ(graph->recorder().mode(), ExecutionMode::ForceReference);
        }

        TEST_F(Small_RecordingGraph, TheGraphRunsAgainAfterItWasEmptied)
        {
            const std::unique_ptr<RecordingGraph> graph = make_graph();
            record_nothing(*graph);
            ASSERT_TRUE(graph->run().has_value());

            record_nothing(*graph);
            EXPECT_EQ(graph->recorded_count(), std::size_t{1});
            EXPECT_TRUE(graph->run().has_value());
        }

        TEST_F(Small_RecordingGraph, ALabelledGraphRecordsAndRunsAsAnUnlabelledOneDoes)
        {
            const std::unique_ptr<RecordingGraph> graph = make_graph();
            graph->set_label("scenario");
            record_nothing(*graph);

            EXPECT_EQ(graph->recorded_count(), std::size_t{1});
            EXPECT_TRUE(graph->run().has_value());
            EXPECT_EQ(graph->recorded_count(), std::size_t{0});
        }

        /** @brief Runs the RecordingGraph tests that guard the release of late-count storage. */
        class Small_Regression_RecordingGraph : public Small_RecordingGraph
        {

        };

        /**
         * @brief Tests that after a run the graph holds no late-count storage, so dropping every
         *        tensor frees it.
         */
        TEST_F(Small_Regression_RecordingGraph, ARunLetsGoOfTheStoragesOfItsLateCountTasks)
        {
            const std::unique_ptr<RecordingGraph> graph = make_graph();
            Tensor out = engine.create_tensor({4}, Core::DataType::Float32);
            Tensor count = engine.create_tensor({1}, Core::DataType::Int32,
                                                SushiRuntime::Residency::Host);
            count.data_as<std::int32_t>()[0] = 2;

            record_late_task(*graph, out, count);
            EXPECT_GT(out.storage->ref_count(), 1);

            ASSERT_TRUE(graph->run().has_value());
            EXPECT_EQ(out.storage->ref_count(), 1);
            EXPECT_EQ(count.storage->ref_count(), 1);
        }

        /** @brief Tests that a discard lets go of the storages of its late-count tasks. */
        TEST_F(Small_Regression_RecordingGraph, ADiscardLetsGoOfTheStoragesOfItsLateCountTasks)
        {
            const std::unique_ptr<RecordingGraph> graph = make_graph();
            Tensor out = engine.create_tensor({4}, Core::DataType::Float32);
            Tensor count = engine.create_tensor({1}, Core::DataType::Int32,
                                                SushiRuntime::Residency::Host);
            count.data_as<std::int32_t>()[0] = 2;

            record_late_task(*graph, out, count);
            EXPECT_EQ(graph->discard(), std::size_t{1});
            EXPECT_EQ(out.storage->ref_count(), 1);
            EXPECT_EQ(count.storage->ref_count(), 1);
        }
    } // namespace Test
} // namespace SushiBLAS
