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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file test_task_handle.cpp
 * @brief Tests TaskHandle: validity, equality and the handle of a call that records nothing.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine/task_handle.hpp>
#include "test_utils.hpp"
#include "graph_probe.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Builds one engine per test for the record calls whose handles are compared. */
        class Unit_TaskHandleTest : public SushiBLASTest
        {
            protected:
                /** @brief Returns a FLOAT32 vector of four elements allocated by engine. */
                [[nodiscard]] static Tensor vector_of_four(Engine& engine)
                {
                    return engine.create_tensor({4}, Core::DataType::Float32);
                }
        };

        /** @brief Tests that a default handle names no task and equals another default. */
        TEST_F(Unit_TaskHandleTest, DefaultHandleNamesNoTask)
        {
            EXPECT_FALSE(TaskHandle().is_valid());
            EXPECT_TRUE(TaskHandle() == TaskHandle());
            EXPECT_FALSE(TaskHandle() != TaskHandle());
        }

        /** @brief Tests that a call which records a task returns a valid handle. */
        TEST_F(Unit_TaskHandleTest, ARecordCallReturnsAValidHandle)
        {
            auto a = vector_of_four(*engine);
            auto b = vector_of_four(*engine);
            auto c = vector_of_four(*engine);

            EXPECT_TRUE(SushiBLAS::ElementwiseOps(*engine).add(a, b, c).is_valid());
        }

        /** @brief Tests that two record calls on one recorder return unequal handles. */
        TEST_F(Unit_TaskHandleTest, TwoRecordCallsReturnDifferentHandles)
        {
            auto a = vector_of_four(*engine);
            auto b = vector_of_four(*engine);
            auto c = vector_of_four(*engine);

            const TaskHandle first = SushiBLAS::ElementwiseOps(*engine).add(a, b, c);
            const TaskHandle second = SushiBLAS::ElementwiseOps(*engine).add(a, b, c);

            EXPECT_TRUE(first != second);
            EXPECT_FALSE(first == second);
        }

        /** @brief Tests that a copy of a handle equals the handle it was copied from. */
        TEST_F(Unit_TaskHandleTest, ACopyComparesEqual)
        {
            auto a = vector_of_four(*engine);
            auto b = vector_of_four(*engine);
            auto c = vector_of_four(*engine);

            const TaskHandle recorded = SushiBLAS::ElementwiseOps(*engine).add(a, b, c);
            const TaskHandle copy = recorded;

            EXPECT_TRUE(copy.is_valid());
            EXPECT_TRUE(copy == recorded);
            EXPECT_FALSE(copy != recorded);
        }

        /** @brief Tests that a call over empty tensors records no node and returns no task. */
        TEST_F(Unit_TaskHandleTest, ANoOpRecordReturnsTheHandleOfNoTask)
        {
            GraphProbe probe;

            auto a = engine->create_tensor({0, 3}, Core::DataType::Float32);
            auto b = engine->create_tensor({0, 3}, Core::DataType::Float32);
            auto c = engine->create_tensor({0, 3}, Core::DataType::Float32);

            const TaskHandle handle = SushiBLAS::ElementwiseOps(probe.recorder()).add(a, b, c);

            EXPECT_FALSE(handle.is_valid());
            EXPECT_TRUE(handle == TaskHandle());
            EXPECT_EQ(0u, probe.take_recorded_count());
        }

        /** @brief Tests that the first handles of two engines compare unequal. */
        TEST_F(Unit_TaskHandleTest, HandlesOfTwoRecordersDiffer)
        {
            Engine other(TestEnvironment::runtime(), Core::Layout::RowMajor);

            auto a = vector_of_four(*engine);
            auto b = vector_of_four(*engine);
            auto c = vector_of_four(*engine);
            auto other_a = vector_of_four(other);
            auto other_b = vector_of_four(other);
            auto other_c = vector_of_four(other);

            const TaskHandle first = SushiBLAS::ElementwiseOps(*engine).add(a, b, c);
            const TaskHandle second =
                SushiBLAS::ElementwiseOps(other).add(other_a, other_b, other_c);

            EXPECT_TRUE(first.is_valid());
            EXPECT_TRUE(second.is_valid());
            EXPECT_TRUE(first != second);
        }
    } // namespace Test
} // namespace SushiBLAS
