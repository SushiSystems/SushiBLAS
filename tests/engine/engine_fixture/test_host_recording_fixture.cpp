/**************************************************************************/
/* test_host_recording_fixture.cpp                                        */
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
 * @file test_host_recording_fixture.cpp
 * @brief Tests that HostRecordingFixture records on the host sink of an engine, runs the bodies and names a failed task.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/engine_fixture/host_recording_fixture.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Records a task named name whose reference half runs work and whose SYCL half does nothing. */
            template<typename Work>
            void record_reference_task(TaskRecorder& recorder, const char* name, Work work)
            {
                TaskDescription description;
                description.name = name;
                recorder.record_pair(
                    description, OperandSet(), "HostEngineTest",
                    [work](const KernelPairDispatcher& pair,
                           const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        return pair.run(deps, work,
                                        [](sycl::queue&, const std::vector<sycl::event>&)
                                        {
                                            return sycl::event();
                                        });
                    });
            }
        } // namespace

        /** @brief Gives each test the host engine and recorder of HostEngineTest. */
        class Small_HostRecordingFixture : public HostEngineTest
        {
        };

        TEST_F(Small_HostRecordingFixture, ARecordedTaskRunsOnRunSucceeds)
        {
            int ran = 0;
            int* ran_out = &ran;
            record_reference_task(recorder(), "host.fixture", [ran_out]() { *ran_out = 1; });
            ASSERT_EQ(recording->recorded_count(), std::size_t{1});

            EXPECT_TRUE(recording->run_succeeds());

            EXPECT_EQ(ran, 1);
            EXPECT_TRUE(recording->is_empty());
        }

        TEST_F(Small_HostRecordingFixture, AFailingTaskIsNamedByRun)
        {
            record_reference_task(recorder(), "host.fixture_failure",
                                  []() { throw std::runtime_error("boom"); });

            const std::optional<HostTaskFailure> failure = recording->run();

            ASSERT_TRUE(failure.has_value());
            EXPECT_EQ(failure->name, "host.fixture_failure");
            EXPECT_EQ(failure->message, "boom");
        }

        TEST_F(Small_HostRecordingFixture, DiscardDropsTheRecordedTasksWithoutRunningThem)
        {
            int ran = 0;
            int* ran_out = &ran;
            record_reference_task(recorder(), "host.fixture", [ran_out]() { *ran_out = 1; });

            EXPECT_EQ(recording->discard(), std::size_t{1});
            EXPECT_TRUE(recording->run_succeeds());

            EXPECT_EQ(ran, 0);
        }
    } // namespace Test
} // namespace SushiBLAS
