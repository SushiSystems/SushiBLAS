/**************************************************************************/
/* test_graph_task_sink.cpp                                               */
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
 * @file test_graph_task_sink.cpp
 * @brief Tests that GraphTaskSink adds one node to its recorder for each task.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <sycl/sycl.hpp>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/graph_task_sink.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include "support/task_sink_subjects.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Builds a graph and an adapter over its recorder. */
        class Medium_GraphTaskSinkTest : public ::testing::Test
        {
            protected:
                /// @brief The graph the adapter records on.
                SushiRuntime::Graph graph = TestEnvironment::runtime().graph();

                /// @brief The adapter under test.
                GraphTaskSink sink{graph.recorder()};

                /// @brief Makes the tensor the tasks write.
                RuntimeTensorSubject tensors;
        };

        /** @brief Tests that a task adds exactly one node to the recorder. */
        TEST_F(Medium_GraphTaskSinkTest, ATaskAddsOneNode)
        {
            const Tensor target = sink_target_of(tensors);
            TaskDescription description;
            description.name = "graph.one_node";
            OperandSet operands;
            operands.writes(target);
            ASSERT_EQ(sink.node_count(), std::size_t{0});

            sink.add_pair(description, operands, ExecutionMode::Auto, std::string(),
                          [](const KernelPairDispatcher&,
                             const std::vector<sycl::event>&) -> sycl::event
                          {
                              return sycl::event();
                          });

            EXPECT_EQ(sink.node_count(), std::size_t{1});
        }
    } // namespace Test
} // namespace SushiBLAS
