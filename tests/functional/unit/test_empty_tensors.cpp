/**************************************************************************/
/* test_empty_tensors.cpp                                                 */
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
 * @file test_empty_tensors.cpp
 * @brief Tests that elementwise, non-linear and logic ops on empty tensors record nothing.
 * @author Mustafa Garip
 */

#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Runs the empty-operand cases.
        class Unit_EmptyTensorTest : public SushiBLASTest
        {
            protected:
                /** @brief Returns a FLOAT32 tensor of shape {0, 3} with no storage. */
                static Tensor empty_tensor()
                {
                    return Tensor(SushiRuntime::sushi_ptr<Storage>(), {0, 3});
                }

                /** @brief Returns whether the graph holds no recorded task, and clears it. */
                bool graph_is_empty()
                {
                    return engine->get_graph().take_nodes().empty();
                }
        };

        /**
         * @brief Tests that add over empty tensors records no task and does not throw.
         */
        TEST_F(Unit_EmptyTensorTest, AddOfEmptyTensorsIsANoOp)
        {
            Tensor a = empty_tensor();
            Tensor b = empty_tensor();
            Tensor c = empty_tensor();

            EXPECT_NO_THROW(engine->elementwise().add(a, b, c));
            EXPECT_TRUE(graph_is_empty());
        }

        /**
         * @brief Tests that exp of an empty tensor records no task and does not throw.
         */
        TEST_F(Unit_EmptyTensorTest, ExpOfAnEmptyTensorIsANoOp)
        {
            Tensor t = empty_tensor();

            EXPECT_NO_THROW(engine->elementwise().exp(t));
            EXPECT_TRUE(graph_is_empty());
        }

        /**
         * @brief Tests that relu of an empty tensor records no task and does not throw.
         */
        TEST_F(Unit_EmptyTensorTest, ReluOfAnEmptyTensorIsANoOp)
        {
            Tensor t = empty_tensor();

            EXPECT_NO_THROW(engine->nonlinear().relu(t));
            EXPECT_TRUE(graph_is_empty());
        }

        /**
         * @brief Tests that logical_and and equal over empty tensors record no task.
         */
        TEST_F(Unit_EmptyTensorTest, LogicOfEmptyTensorsIsANoOp)
        {
            Tensor a = empty_tensor();
            Tensor b = empty_tensor();
            Tensor r = empty_tensor();

            EXPECT_NO_THROW(engine->logic().logical_and(a, b, r));
            EXPECT_NO_THROW(engine->logic().equal(a, b, r));
            EXPECT_TRUE(graph_is_empty());
        }

        /**
         * @brief Tests that all of a storage-less empty tensor writes the identity 1.
         */
        TEST_F(Unit_EmptyTensorTest, AllOfAnEmptyTensorWritesOne)
        {
            Tensor t = empty_tensor();
            Tensor out = engine->create_tensor({1});
            fill_tensor(out, {0.5f});

            EXPECT_NO_THROW(engine->logic().all(t, out));
            engine->execute().wait();
            verify_tensor(out, {1.0f});
        }

        /**
         * @brief Tests that any of a storage-less empty tensor writes the identity 0.
         */
        TEST_F(Unit_EmptyTensorTest, AnyOfAnEmptyTensorWritesZero)
        {
            Tensor t = empty_tensor();
            Tensor out = engine->create_tensor({1});
            fill_tensor(out, {0.5f});

            EXPECT_NO_THROW(engine->logic().any(t, out));
            engine->execute().wait();
            verify_tensor(out, {0.0f});
        }

        /**
         * @brief Tests that convert between empty tensors records no task.
         */
        TEST_F(Unit_EmptyTensorTest, ConvertOfEmptyTensorsIsANoOp)
        {
            Tensor src = empty_tensor();
            Tensor dst(SushiRuntime::sushi_ptr<Storage>(), {0, 3}, Core::DataType::FLOAT64);

            EXPECT_NO_THROW(engine->elementwise().convert(src, dst));
            EXPECT_TRUE(graph_is_empty());
        }

        /**
         * @brief Tests that empty operands of different shapes still throw.
         */
        TEST_F(Unit_EmptyTensorTest, EmptyOperandsMustStillAgreeInShape)
        {
            Tensor a = empty_tensor();
            Tensor b(SushiRuntime::sushi_ptr<Storage>(), {3, 0});
            Tensor c = empty_tensor();

            EXPECT_THROW(engine->elementwise().add(a, b, c), std::invalid_argument);
            EXPECT_TRUE(graph_is_empty());
        }
    } // namespace Test
} // namespace SushiBLAS
