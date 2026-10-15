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
 * @brief Tests what elementwise, non-linear, logic and reduction ops do with an empty tensor.
 * @author Mustafa Garip
 */

#include <vector>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>

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

                /** @brief Returns whether the probe holds no recorded task, and clears it. */
                bool graph_is_empty()
                {
                    return probe.take_recorded_count() == 0;
                }

                /// @brief The graph the tests of this suite record on and read.
                GraphProbe probe;
        };

        /**
         * @brief Tests that add over empty tensors records no task and does not throw.
         */
        TEST_F(Unit_EmptyTensorTest, AddOfEmptyTensorsIsANoOp)
        {
            Tensor a = empty_tensor();
            Tensor b = empty_tensor();
            Tensor c = empty_tensor();

            EXPECT_NO_THROW(SushiBLAS::ElementwiseOps(probe.recorder()).add(a, b, c));
            EXPECT_TRUE(graph_is_empty());
        }

        /**
         * @brief Tests that exp of an empty tensor records no task and does not throw.
         */
        TEST_F(Unit_EmptyTensorTest, ExpOfAnEmptyTensorIsANoOp)
        {
            Tensor t = empty_tensor();

            EXPECT_NO_THROW(SushiBLAS::ElementwiseOps(probe.recorder()).exp(t));
            EXPECT_TRUE(graph_is_empty());
        }

        /**
         * @brief Tests that relu of an empty tensor records no task and does not throw.
         */
        TEST_F(Unit_EmptyTensorTest, ReLUOfAnEmptyTensorIsANoOp)
        {
            Tensor t = empty_tensor();

            EXPECT_NO_THROW(SushiBLAS::NonLinearOps(probe.recorder()).relu(t));
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

            EXPECT_NO_THROW(SushiBLAS::LogicOps(probe.recorder()).logical_and(a, b, r));
            EXPECT_NO_THROW(SushiBLAS::LogicOps(probe.recorder()).equal(a, b, r));
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

            EXPECT_NO_THROW(SushiBLAS::LogicOps(*engine).all(t, out));
            engine->execute().value();
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

            EXPECT_NO_THROW(SushiBLAS::LogicOps(*engine).any(t, out));
            engine->execute().value();
            verify_tensor(out, {0.0f});
        }

        /**
         * @brief Tests that convert between empty tensors records no task.
         */
        TEST_F(Unit_EmptyTensorTest, ConvertOfEmptyTensorsIsANoOp)
        {
            Tensor src = empty_tensor();
            Tensor dst(SushiRuntime::sushi_ptr<Storage>(), {0, 3}, Core::DataType::Float64);

            EXPECT_NO_THROW(SushiBLAS::ElementwiseOps(probe.recorder()).convert(src, dst));
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

            EXPECT_THROW(
                SushiBLAS::ElementwiseOps(probe.recorder()).add(a, b, c),
                std::invalid_argument);
            EXPECT_TRUE(graph_is_empty());
        }

        /**
         * @brief Tests that the five reductions with an identity write it for an empty tensor.
         */
        TEST_F(Unit_EmptyTensorTest, IdentityReductionsOfNothingWriteTheirIdentity)
        {
            Tensor t = empty_tensor();
            SushiBLAS::ReductionOps reductions(*engine);

            Tensor sum = engine->create_tensor({1});
            Tensor asum = engine->create_tensor({1});
            Tensor nrm2 = engine->create_tensor({1});
            Tensor prod = engine->create_tensor({1});
            Tensor count = engine->create_tensor({1}, Core::DataType::Int64);

            // Each result starts from a value that is not its identity.
            fill_tensor(sum, {0.5f});
            fill_tensor(asum, {0.5f});
            fill_tensor(nrm2, {0.5f});
            fill_tensor(prod, {0.5f});
            fill_tensor(count, {7.0f});

            EXPECT_NO_THROW(reductions.sum(t, sum));
            EXPECT_NO_THROW(reductions.asum(t, asum));
            EXPECT_NO_THROW(reductions.nrm2(t, nrm2));
            EXPECT_NO_THROW(reductions.prod(t, prod));
            EXPECT_NO_THROW(reductions.count_nonfinite(t, count));
            engine->execute().value();

            verify_tensor(sum, {0.0f});
            verify_tensor(asum, {0.0f});
            verify_tensor(nrm2, {0.0f});
            verify_tensor(prod, {1.0f});
            verify_tensor(count, {0.0f});
        }

        /**
         * @brief Tests that every reduction without an identity refuses an empty tensor.
         */
        TEST_F(Unit_EmptyTensorTest, OrderedReductionsOfNothingAreRefused)
        {
            Tensor t = empty_tensor();
            Tensor image(SushiRuntime::sushi_ptr<Storage>(), {0, 2, 2, 3});
            SushiBLAS::ReductionOps reductions(probe.recorder());

            Tensor scalar = engine->create_tensor({1});
            Tensor index = engine->create_tensor({1}, Core::DataType::Int64);
            Tensor per_column = engine->create_tensor({3});
            Tensor per_channel = engine->create_tensor({1, 3});

            EXPECT_THROW(reductions.mean(t, scalar), std::invalid_argument);
            EXPECT_THROW(reductions.max(t, scalar), std::invalid_argument);
            EXPECT_THROW(reductions.min(t, scalar), std::invalid_argument);
            EXPECT_THROW(reductions.var(t, scalar), std::invalid_argument);
            EXPECT_THROW(reductions.std(t, scalar), std::invalid_argument);
            EXPECT_THROW(reductions.logsumexp(t, scalar), std::invalid_argument);
            EXPECT_THROW(reductions.argmax(t, index), std::invalid_argument);
            EXPECT_THROW(reductions.argmin(t, index), std::invalid_argument);
            EXPECT_THROW(reductions.max_along(t, 0, per_column), std::invalid_argument);
            EXPECT_THROW(reductions.min_along(t, 0, per_column), std::invalid_argument);
            EXPECT_THROW(reductions.mean_spatial(image, per_channel), std::invalid_argument);
            EXPECT_TRUE(graph_is_empty());
        }

        /**
         * @brief Tests sum_along over an axis of length zero and into a result of length zero.
         */
        TEST_F(Unit_EmptyTensorTest, SumAlongOfAnEmptyAxisWritesZeros)
        {
            // Folding axis 0 of a [0, 5] tensor leaves five sums of nothing.
            Tensor no_rows(SushiRuntime::sushi_ptr<Storage>(), {0, 5});
            Tensor sums = engine->create_tensor({5});
            fill_tensor(sums, std::vector<float>(5, 0.5f));

            EXPECT_NO_THROW(SushiBLAS::ReductionOps(*engine).sum_along(no_rows, 0, sums));
            engine->execute().value();
            verify_tensor(sums, std::vector<float>(5, 0.0f));

            // Folding axis 0 of a [5, 0] tensor leaves no position to write.
            Tensor no_columns(SushiRuntime::sushi_ptr<Storage>(), {5, 0});
            Tensor nothing(SushiRuntime::sushi_ptr<Storage>(), {0});

            EXPECT_NO_THROW(
                SushiBLAS::ReductionOps(probe.recorder()).sum_along(no_columns, 0, nothing));
            EXPECT_TRUE(graph_is_empty());
        }
    } // namespace Test
} // namespace SushiBLAS
