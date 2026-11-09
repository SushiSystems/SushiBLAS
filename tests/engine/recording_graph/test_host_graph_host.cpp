/**************************************************************************/
/* test_host_graph_host.cpp                                               */
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
 * @file test_host_graph_host.cpp
 * @brief Tests that HostGraphHost counts, runs and drops the tasks of its sink, names the task that failed and admits the reference mode only.
 * @author Mustafa Garip
 * @see tests/engine/recording_graph/README.md
 */

#include <string>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/recording_graph/host_graph_host.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Records a task named name whose host body throws when fail is set. */
            void record_task(TaskRecorder& recorder, const char* name, bool fail)
            {
                TaskDescription description;
                description.name = name;
                recorder.record_pair(description, OperandSet(), "test",
                                     [fail](const KernelPairDispatcher& dispatcher,
                                            const std::vector<sycl::event>& deps)
                                     {
                                         return dispatcher.run(deps,
                                             [fail]()
                                             {
                                                 if (fail)
                                                     throw std::runtime_error("body failed");
                                             },
                                             [](sycl::queue&, const std::vector<sycl::event>&)
                                             {
                                                 return sycl::event();
                                             });
                                     });
            }

            /** @brief Builds a host and a recorder over its sink for each test. */
            class Small_HostGraphHost : public ::testing::Test
            {
                protected:
                    /// @brief The host under test.
                    HostGraphHost host;

                    /// @brief The recorder over the sink of host.
                    TaskRecorder recorder{host.sink()};
            };
        } // namespace

        TEST_F(Small_HostGraphHost, CountsTheTasksRecordedOnItsSink)
        {
            record_task(recorder, "a", false);
            record_task(recorder, "b", false);

            EXPECT_EQ(host.node_count(), std::size_t{2});
        }

        TEST_F(Small_HostGraphHost, ARunKeepsTheTasksAndClearDropsThem)
        {
            record_task(recorder, "a", false);

            EXPECT_TRUE(host.run().has_value());
            EXPECT_EQ(host.node_count(), std::size_t{1});

            host.clear();
            EXPECT_EQ(host.node_count(), std::size_t{0});
        }

        TEST_F(Small_HostGraphHost, ARunNamesTheTaskThatFailed)
        {
            record_task(recorder, "first", false);
            record_task(recorder, "second", true);

            const auto ran = host.run();

            ASSERT_FALSE(ran.has_value());
            EXPECT_EQ(ran.error().code, SushiRuntime::ErrorCode::KernelFailed);
            EXPECT_NE(ran.error().message.find("task 1 'second'"), std::string::npos);
            EXPECT_NE(ran.error().message.find("body failed"), std::string::npos);
        }

        TEST_F(Small_HostGraphHost, HasNoPlan)
        {
            const auto plan = host.plan();

            ASSERT_FALSE(plan.has_value());
            EXPECT_EQ(plan.error().code, SushiRuntime::ErrorCode::Unsupported);
        }

        TEST_F(Small_HostGraphHost, TakesNoLateCountNodeAndHoldsNoLateCountStorage)
        {
            EXPECT_EQ(host.late_sink(), nullptr);
            EXPECT_EQ(host.late_counts(), nullptr);
            EXPECT_FALSE(host.has_refusal());
        }

        TEST_F(Small_HostGraphHost, AdmitsTheReferenceModeOnly)
        {
            EXPECT_TRUE(host.sink().admits(ExecutionMode::ForceReference));
            EXPECT_FALSE(host.sink().admits(ExecutionMode::ForceSYCL));
        }
    } // namespace Test
} // namespace SushiBLAS
