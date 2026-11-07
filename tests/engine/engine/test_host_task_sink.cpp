/**************************************************************************/
/* test_host_task_sink.cpp                                                */
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
 * @file test_host_task_sink.cpp
 * @brief Tests the parts of HostTaskSink that the TaskSink contract leaves out.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <optional>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/host_task_sink.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include "support/task_sink_subjects.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Adds a task named name whose body adds one to *counter. */
            void add_counting_task(HostTaskSink& sink, const char* name, int* counter)
            {
                TaskDescription description;
                description.name = name;
                sink.add_pair(description, OperandSet(), ExecutionMode::ForceReference,
                              std::string(),
                              [counter](const KernelPairDispatcher&,
                                        const std::vector<sycl::event>&) -> sycl::event
                              {
                                  ++*counter;
                                  return sycl::event();
                              });
            }

            /** @brief Adds a task named name whose body throws message. */
            void add_throwing_task(HostTaskSink& sink, const char* name, const char* message)
            {
                TaskDescription description;
                description.name = name;
                sink.add_pair(description, OperandSet(), ExecutionMode::ForceReference,
                              std::string(),
                              [message](const KernelPairDispatcher&,
                                        const std::vector<sycl::event>&) -> sycl::event
                              {
                                  throw std::runtime_error(message);
                              });
            }
        } // namespace

        /** @brief Tests that the host sink admits ForceReference alone and has no graph. */
        TEST(Small_HostTaskSink, ItAdmitsForceReferenceOnlyAndHasNoGraph)
        {
            HostTaskSink sink;

            EXPECT_TRUE(sink.admits(ExecutionMode::ForceReference));
            EXPECT_FALSE(sink.admits(ExecutionMode::Auto));
            EXPECT_FALSE(sink.admits(ExecutionMode::ForceSYCL));
            EXPECT_EQ(sink.graph(), nullptr);
        }

        /** @brief Tests that a run of nothing succeeds. */
        TEST(Small_HostTaskSink, AnEmptySinkRunsSuccessfully)
        {
            HostTaskSink sink;

            EXPECT_FALSE(sink.run().has_value());
            EXPECT_EQ(sink.recorded_count(), std::size_t{0});
        }

        /** @brief Tests that a run drops the bodies it ran, so the next run runs none. */
        TEST(Small_HostTaskSink, ARunDropsTheBodiesItRan)
        {
            HostTaskSink sink;
            int counter = 0;
            add_counting_task(sink, "host.count", &counter);
            ASSERT_EQ(sink.recorded_count(), std::size_t{1});

            EXPECT_FALSE(sink.run().has_value());
            EXPECT_FALSE(sink.run().has_value());

            EXPECT_EQ(counter, 1);
            EXPECT_EQ(sink.recorded_count(), std::size_t{0});
        }

        /** @brief Tests that the failure names the first failing task by place, name and message. */
        TEST(Small_HostTaskSink, TheFirstFailureNamesItsIndexNameAndMessage)
        {
            HostTaskSink sink;
            int counter = 0;
            add_counting_task(sink, "host.before", &counter);
            add_throwing_task(sink, "host.first_failure", "first");
            add_throwing_task(sink, "host.second_failure", "second");

            const std::optional<HostTaskFailure> failure = sink.run();

            ASSERT_TRUE(failure.has_value());
            EXPECT_EQ(failure->index, std::size_t{1});
            EXPECT_EQ(failure->name, "host.first_failure");
            EXPECT_EQ(failure->message, "first");
            EXPECT_EQ(counter, 1);
            EXPECT_EQ(sink.recorded_count(), std::size_t{0});
        }

        /** @brief Tests that discard drops the bodies without running them. */
        TEST(Small_HostTaskSink, DiscardDropsTheBodiesUnrun)
        {
            HostTaskSink sink;
            int counter = 0;
            add_counting_task(sink, "host.count", &counter);
            add_counting_task(sink, "host.count_again", &counter);

            EXPECT_EQ(sink.discard(), std::size_t{2});
            EXPECT_FALSE(sink.run().has_value());
            EXPECT_EQ(counter, 0);
        }

        /** @brief Tests that a body which throws something that is not a std::exception is reported. */
        TEST(Small_HostTaskSink, ABodyThatThrowsAnIntegerIsReported)
        {
            HostTaskSink sink;
            TaskDescription description;
            description.name = "host.integer";
            sink.add_pair(description, OperandSet(), ExecutionMode::ForceReference, std::string(),
                          [](const KernelPairDispatcher&,
                             const std::vector<sycl::event>&) -> sycl::event
                          {
                              throw 7;
                          });

            const std::optional<HostTaskFailure> failure = sink.run();

            ASSERT_TRUE(failure.has_value());
            EXPECT_EQ(failure->name, "host.integer");
            EXPECT_FALSE(failure->message.empty());
        }
    } // namespace Test
} // namespace SushiBLAS
