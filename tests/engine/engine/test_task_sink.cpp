/**************************************************************************/
/* test_task_sink.cpp                                                     */
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
 * @file test_task_sink.cpp
 * @brief Tests that the host and graph adapters of TaskSink keep the TaskSink contract, and what a
 *        recorder on a host sink and a dispatcher with no queue refuse.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <sycl/sycl.hpp>
#include <gtest/gtest.h>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/host_task_sink.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include "support/task_sink_contract.hpp"
#include "support/task_sink_subjects.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the TaskSink contract against a subject on the host. */
        template<typename Subject>
        class Small_TaskSinkContract : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        /** @brief Runs the TaskSink contract against a subject on the shared runtime. */
        template<typename Subject>
        class Medium_TaskSinkContract : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        using HostSinkSubjects = ::testing::Types<HostSinkSubject>;
        using GraphSinkSubjects = ::testing::Types<GraphSinkSubject>;
        SB_TYPED_TEST_SUITE(Small_TaskSinkContract, HostSinkSubjects);
        SB_TYPED_TEST_SUITE(Medium_TaskSinkContract, GraphSinkSubjects);

        TYPED_TEST(Small_TaskSinkContract, RecordOrderIsRunOrder)
        {
            TaskSinkContract::expect_record_order_is_run_order(this->subject);
        }

        TYPED_TEST(Small_TaskSinkContract, TheReferenceHalfRunsUnderForceReference)
        {
            TaskSinkContract::expect_reference_half_runs_under_force_reference(this->subject);
        }

        TYPED_TEST(Small_TaskSinkContract, AFailingBodyStopsTheRun)
        {
            TaskSinkContract::expect_failing_body_stops_the_run(this->subject);
        }

        TYPED_TEST(Small_TaskSinkContract, TheQueueFollowsTheAdapter)
        {
            TaskSinkContract::expect_queue_follows_the_adapter(this->subject);
        }

        TYPED_TEST(Medium_TaskSinkContract, RecordOrderIsRunOrder)
        {
            TaskSinkContract::expect_record_order_is_run_order(this->subject);
        }

        TYPED_TEST(Medium_TaskSinkContract, TheReferenceHalfRunsUnderForceReference)
        {
            TaskSinkContract::expect_reference_half_runs_under_force_reference(this->subject);
        }

        TYPED_TEST(Medium_TaskSinkContract, AFailingBodyStopsTheRun)
        {
            TaskSinkContract::expect_failing_body_stops_the_run(this->subject);
        }

        TYPED_TEST(Medium_TaskSinkContract, TheQueueFollowsTheAdapter)
        {
            TaskSinkContract::expect_queue_follows_the_adapter(this->subject);
        }

        /** @brief Tests that a dispatcher built with no queue says so and refuses queue(). */
        TEST(Small_KernelPairDispatcherQueue, ADispatcherWithNoQueueThrowsOnQueue)
        {
            const KernelPairDispatcher dispatcher(ExecutionMode::ForceReference);

            EXPECT_FALSE(dispatcher.has_queue());
            EXPECT_THROW((void)dispatcher.queue(), UnsupportedExecution);
        }

        /** @brief Tests that ForceReference runs the reference half with no queue. */
        TEST(Small_KernelPairDispatcherQueue, ForceReferenceRunsTheReferenceWithNoQueue)
        {
            const KernelPairDispatcher dispatcher(ExecutionMode::ForceReference);
            int ran = 0;

            dispatcher.run({}, [&ran]() { ran = 1; },
                           [&ran](sycl::queue&, const std::vector<sycl::event>&)
                           {
                               ran = 2;
                               return sycl::event();
                           });

            EXPECT_EQ(ran, 1);
        }

        namespace
        {
            /** @brief Expects a dispatcher in mode with no queue to refuse to run either half. */
            void expect_no_half_runs_without_a_queue(ExecutionMode mode)
            {
                const KernelPairDispatcher dispatcher(mode);
                int ran = 0;

                EXPECT_THROW(
                    dispatcher.run({}, [&ran]() { ran = 1; },
                                   [&ran](sycl::queue&, const std::vector<sycl::event>&)
                                   {
                                       ran = 2;
                                       return sycl::event();
                                   }),
                    UnsupportedExecution);
                EXPECT_EQ(ran, 0);
            }
        } // namespace

        /** @brief Tests that Auto needs the queue and throws when there is none. */
        TEST(Small_KernelPairDispatcherQueue, AutoWithNoQueueThrows)
        {
            expect_no_half_runs_without_a_queue(ExecutionMode::Auto);
        }

        /** @brief Tests that ForceSYCL needs the queue and throws when there is none. */
        TEST(Small_KernelPairDispatcherQueue, ForceSYCLWithNoQueueThrows)
        {
            expect_no_half_runs_without_a_queue(ExecutionMode::ForceSYCL);
        }

        /** @brief Tests that a recorder on a host sink refuses every mode but ForceReference. */
        TEST(Small_HostTaskRecorder, ARecorderOnAHostSinkRecordsForceReferenceOnly)
        {
            HostTaskSink sink;

            EXPECT_THROW(TaskRecorder(sink, nullptr, nullptr, ExecutionMode::Auto),
                         UnsupportedExecution);
            EXPECT_THROW(TaskRecorder(sink, nullptr, nullptr, ExecutionMode::ForceSYCL),
                         UnsupportedExecution);

            TaskRecorder recorder(sink);
            EXPECT_EQ(recorder.mode(), ExecutionMode::ForceReference);
            EXPECT_THROW(recorder.set_mode(ExecutionMode::Auto), UnsupportedExecution);
            EXPECT_EQ(recorder.mode(), ExecutionMode::ForceReference);
            EXPECT_NO_THROW(recorder.set_mode(ExecutionMode::ForceReference));
        }

        /** @brief Tests that a recorder built without a pool refuses to lend one. */
        TEST(Small_HostTaskRecorder, ARecorderWithNoPoolRefusesScratch)
        {
            HostTaskSink sink;
            TaskRecorder recorder(sink);

            EXPECT_THROW((void)recorder.scratch_pool(), UnsupportedExecution);
        }

        /** @brief Tests that a body of the old shape fails on a host sink, which has no queue to give it. */
        TEST(Small_HostTaskRecorder, ABodyOfTheOldShapeIsReportedAsFailedOnAHostSink)
        {
            HostSinkSubject subject;
            bool ran = false;
            bool* ran_out = &ran;
            TaskDescription description;
            description.name = "sink.old_shape";
            OperandSet operands;
            operands.writes(subject.target());

            subject.recorder().record_pair(
                description, operands, "TaskSink contract",
                [ran_out](const KernelPairDispatcher&, sycl::queue&,
                          const std::vector<sycl::event>&) -> sycl::event
                {
                    *ran_out = true;
                    return sycl::event();
                });

            const std::string failure = subject.run();

            EXPECT_NE(failure.find("sink.old_shape"), std::string::npos) << failure;
            EXPECT_NE(failure.find("no queue"), std::string::npos) << failure;
            EXPECT_FALSE(ran);
        }

        /** @brief Tests that a body of the old shape gets the queue of the graph adapter. */
        TEST(Medium_GraphTaskRecorder, ABodyOfTheOldShapeGetsTheDispatchersQueue)
        {
            GraphSinkSubject subject;
            bool matched = false;
            bool* matched_out = &matched;
            TaskDescription description;
            description.name = "sink.old_shape";
            OperandSet operands;
            operands.writes(subject.target());

            subject.recorder().record_pair(
                description, operands, "TaskSink contract",
                [matched_out](const KernelPairDispatcher& pair, sycl::queue& q,
                              const std::vector<sycl::event>&) -> sycl::event
                {
                    *matched_out = (&q == &pair.queue());
                    return sycl::event();
                });

            EXPECT_EQ(subject.run(), std::string());
            EXPECT_TRUE(matched);
        }
    } // namespace Test
} // namespace SushiBLAS
