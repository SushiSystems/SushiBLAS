/**************************************************************************/
/* test_late_count_sink.cpp                                               */
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
 * @file test_late_count_sink.cpp
 * @brief Tests that a LateCountSink adds the one node a late record adds and refuses a record
 *        that adds none, naming the task.
 * @author Mustafa Garip
 */

#include <string>
#include <cstddef>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/cost/cost.hpp>
#include <SushiBLAS/engine/graph_task_sink.hpp>
#include <SushiBLAS/engine/late_count_sink.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds a graph and the graph adapter over its recorder, the one LateCountSink adapter. */
        class GraphLateCountSubject
        {
            public:
                /** @brief Returns the port under test. */
                [[nodiscard]] LateCountSink& sink() noexcept { return sink_; }

                /** @brief Returns how many nodes the graph holds. */
                [[nodiscard]] std::size_t node_count() const noexcept { return sink_.node_count(); }

            private:
                /// @brief The graph the adapter records on.
                SushiRuntime::Graph graph_ = TestEnvironment::runtime().graph();

                /// @brief The adapter under test.
                GraphTaskSink sink_{graph_.recorder()};
        };

        /** @brief Runs the LateCountSink contract against an adapter on the shared runtime. */
        template<typename Subject>
        class Medium_LateCountSinkContract : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        /// @brief The LateCountSink adapters; a late-count kernel has no host form, so there is no host one.
        using LateCountSinkSubjects = ::testing::Types<GraphLateCountSubject>;

        SB_TYPED_TEST_SUITE(Medium_LateCountSinkContract, LateCountSinkSubjects);

        /** @brief Tests that a late record that adds a node leaves exactly one more node. */
        TYPED_TEST(Medium_LateCountSinkContract, ALateRecordAddsOneNode)
        {
            TaskDescription description;
            description.name = "late.one_node";

            this->subject.sink().add_late_node(
                description, Cost::uncosted(UncostedReason::DataDependent), std::string(),
                [](SushiRuntime::Recorder& recorder, const SushiRuntime::NodeOptions& options)
                {
                    recorder.host(SushiRuntime::Access{}, [] {}, options);
                });

            EXPECT_EQ(this->subject.node_count(), std::size_t{1});
        }

        /** @brief Tests that a late record that adds no node is refused with the task's name. */
        TYPED_TEST(Medium_LateCountSinkContract, ARecordThatAddsNoNodeIsRefusedNamingTheTask)
        {
            TaskDescription description;
            description.name = "late.no_node";
            std::string message;

            try
            {
                this->subject.sink().add_late_node(
                    description, Cost::uncosted(UncostedReason::DataDependent), std::string(),
                    [](SushiRuntime::Recorder&, const SushiRuntime::NodeOptions&) {});
            }
            catch (const std::runtime_error& refusal)
            {
                message = refusal.what();
            }

            EXPECT_NE(message.find("late.no_node"), std::string::npos) << message;
        }
    } // namespace Test
} // namespace SushiBLAS
