/**************************************************************************/
/* test_task_handle.cpp                                                   */
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
 * @file test_task_handle.cpp
 * @brief Tests TaskHandle: validity, equality and the handle of a call that records nothing.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Builds one engine and one graph per test for the record calls whose handles are compared. */
        class Medium_TaskHandleTest : public SushiBLASTest
        {
            protected:
                /** @brief Returns a FLOAT32 vector of four elements allocated by engine. */
                [[nodiscard]] static Tensor vector_of_four(Engine& engine)
                {
                    return engine.create_tensor({4}, Core::DataType::Float32);
                }
        };

        namespace
        {
            /** @brief Names a shape for the operands of one add and whether it records a task. */
            struct RecordRow
            {
                const char* name;             /**< The row name a failure shows. */
                std::vector<int64_t> dims;    /**< The shape of every operand. */
                bool records;                 /**< Whether the call records a task. */
            };

            /** @brief Returns the shape that records a task and the empty shape that records none. */
            std::vector<RecordRow> record_rows()
            {
                return {{"AFourElementVectorRecordsATask", {4}, true},
                        {"EmptyTensorsRecordNothing", {0, 3}, false}};
            }

            /** @brief Names whether the second handle comes from the first recorder or from another. */
            struct RecorderRow
            {
                const char* name;      /**< The row name a failure shows. */
                bool other_recorder;   /**< Whether the second call records on another recorder. */
            };

            /** @brief Returns the row of one recorder and the row of two. */
            std::vector<RecorderRow> recorder_rows()
            {
                return {{"TwoCallsOnOneRecorder", false}, {"OneCallOnEachOfTwoRecorders", true}};
            }
        } // namespace

        /** @brief Runs one row of record_rows(). */
        class Medium_TaskHandleRecord : public Medium_TaskHandleTest,
                                        public ::testing::WithParamInterface<RecordRow>
        {

        };

        TEST_P(Medium_TaskHandleRecord, AHandleIsValidExactlyWhenTheCallRecordedATask)
        {
            auto a = engine->create_tensor(GetParam().dims, Core::DataType::Float32);
            auto b = engine->create_tensor(GetParam().dims, Core::DataType::Float32);
            auto c = engine->create_tensor(GetParam().dims, Core::DataType::Float32);

            const TaskHandle handle = SushiBLAS::ElementwiseOps(recorder()).add(a, b, c);

            EXPECT_EQ(handle.is_valid(), GetParam().records);
            EXPECT_EQ(handle == TaskHandle(), !GetParam().records);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_TaskHandleRecord, ::testing::ValuesIn(record_rows()),
            [](const ::testing::TestParamInfo<RecordRow>& info)
            { return std::string(info.param.name); });

        /** @brief Runs one row of recorder_rows(). */
        class Medium_TaskHandleDiffer : public Medium_TaskHandleTest,
                                        public ::testing::WithParamInterface<RecorderRow>
        {

        };

        TEST_P(Medium_TaskHandleDiffer, TwoHandlesFromRecordCallsDiffer)
        {
            RecordingFixture other(TestEnvironment::runtime(), *engine);
            TaskRecorder& second_recorder = GetParam().other_recorder ? other.recorder() : recorder();
            auto a = vector_of_four(*engine);
            auto b = vector_of_four(*engine);
            auto c = vector_of_four(*engine);

            const TaskHandle first = SushiBLAS::ElementwiseOps(recorder()).add(a, b, c);
            const TaskHandle second = SushiBLAS::ElementwiseOps(second_recorder).add(a, b, c);

            EXPECT_TRUE(first.is_valid());
            EXPECT_TRUE(second.is_valid());
            EXPECT_TRUE(first != second);
            EXPECT_FALSE(first == second);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_TaskHandleDiffer, ::testing::ValuesIn(recorder_rows()),
            [](const ::testing::TestParamInfo<RecorderRow>& info)
            { return std::string(info.param.name); });

        /** @brief Tests that a copy of a handle equals the handle it was copied from. */
        TEST_F(Medium_TaskHandleTest, ACopyComparesEqual)
        {
            auto a = vector_of_four(*engine);
            auto b = vector_of_four(*engine);
            auto c = vector_of_four(*engine);

            const TaskHandle recorded = SushiBLAS::ElementwiseOps(recorder()).add(a, b, c);
            const TaskHandle copy = recorded;

            EXPECT_TRUE(copy.is_valid());
            EXPECT_TRUE(copy == recorded);
            EXPECT_FALSE(copy != recorded);
        }

    } // namespace Test
} // namespace SushiBLAS
