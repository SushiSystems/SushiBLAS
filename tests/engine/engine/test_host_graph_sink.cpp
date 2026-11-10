/**************************************************************************/
/* test_host_graph_sink.cpp                                               */
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
 * @file test_host_graph_sink.cpp
 * @brief Tests that HostGraphSink admits every mode and keeps the name, cost and origin of a late-count node without calling its record.
 * @author Mustafa Garip
 */

#include <string>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/cost/cost.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/host_graph_sink.hpp>
#include <SushiBLAS/engine/task_description.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Runs the admission of a HostGraphSink for one execution mode.
        class Small_HostGraphSinkAdmission : public ::testing::TestWithParam<ExecutionMode>
        {

        };

        TEST_P(Small_HostGraphSinkAdmission, EveryModeIsAdmitted)
        {
            const HostGraphSink sink;

            EXPECT_TRUE(sink.admits(GetParam()));
        }

        INSTANTIATE_TEST_SUITE_P(Modes, Small_HostGraphSinkAdmission,
                                 ::testing::Values(ExecutionMode::Auto, ExecutionMode::ForceSYCL,
                                                   ExecutionMode::ForceReference));

        TEST(Small_HostGraphSink, ALateNodeKeepsItsNameCostAndOriginAndItsRecordNeverRuns)
        {
            HostGraphSink sink;
            TaskDescription description;
            description.name = "late.add";
            bool record_ran = false;

            sink.add_late_node(description, Cost::uncosted(UncostedReason::DataDependent), "scope",
                               [&record_ran](SushiRuntime::Recorder&, const SushiRuntime::NodeOptions&)
                               { record_ran = true; });

            ASSERT_EQ(sink.late_nodes().size(), std::size_t{1});
            EXPECT_EQ(sink.late_nodes().front().name, "late.add");
            EXPECT_EQ(sink.late_nodes().front().cost.reason, UncostedReason::DataDependent);
            EXPECT_EQ(sink.late_nodes().front().origin, "scope");
            EXPECT_FALSE(record_ran);
        }
    } // namespace Test
} // namespace SushiBLAS
