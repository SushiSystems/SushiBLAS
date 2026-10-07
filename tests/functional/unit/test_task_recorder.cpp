/**************************************************************************/
/* test_task_recorder.cpp                                                 */
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
 * @file test_task_recorder.cpp
 * @brief Tests TaskRecorder: the graph and pool it is bound to, and what each mode records.
 * @author Mustafa Garip
 */

#include <array>
#include <atomic>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <functional>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/support/spatial_geometry.hpp>
#include "../../common/test_utils.hpp"
#include "../../common/graph_probe.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Builds the tensors and records the call chain the TaskRecorder tests share. */
        class Unit_TaskRecorderTest : public SushiBLASTest
        {
            protected:
                /** @brief Holds the tensors one replay of the three chained view calls writes. */
                struct ChainOutputs
                {
                    Tensor squared; /**< Receives input * input. */
                    Tensor shifted; /**< Receives squared + input. */
                    Tensor total;   /**< Receives the sum of shifted. */
                };

                /** @brief Names one record call of a family that has a SYCL kernel only. */
                struct DeviceOnlyCall
                {
                    const char* family;         /**< The family, for the failure message. */
                    std::function<void()> call; /**< Records one operation of the family. */
                };

                /** @brief Returns a FLOAT32 vector of four elements from the fixture's engine. */
                [[nodiscard]] Tensor vector_of_four()
                {
                    return engine->create_tensor({4}, Core::DataType::FLOAT32);
                }

                /** @brief Returns fresh outputs for one replay over a four-element input. */
                [[nodiscard]] ChainOutputs chain_outputs()
                {
                    return ChainOutputs{vector_of_four(), vector_of_four(),
                                        engine->create_tensor({1}, Core::DataType::FLOAT32)};
                }

                /** @brief Records mul, add and sum on recorder, each reading the one before. */
                static void record_chain(TaskRecorder& recorder, const Tensor& input,
                                         ChainOutputs& outputs)
                {
                    SushiBLAS::ElementwiseOps(recorder).mul(input, input, outputs.squared);
                    SushiBLAS::ElementwiseOps(recorder).add(outputs.squared, input,
                                                            outputs.shifted);
                    SushiBLAS::ReductionOps(recorder).sum(outputs.shifted, outputs.total);
                }

                /** @brief Returns the in-degree of every node, in the order they were recorded. */
                [[nodiscard]] static std::vector<int32_t> in_degrees_of(
                    const std::vector<SushiRuntime::sushi_ptr<SushiRuntime::Graph::Node>>& nodes)
                {
                    std::vector<int32_t> degrees;
                    degrees.reserve(nodes.size());
                    for (const auto& node : nodes)
                        degrees.push_back(node->in_degree.load(std::memory_order_relaxed));
                    return degrees;
                }
        };

        /** @brief Tests that a record call adds its node to the recorder's graph only. */
        TEST_F(Unit_TaskRecorderTest, RecordsOnTheGraphItWasGiven)
        {
            GraphProbe first;
            GraphProbe second;

            auto x = vector_of_four();
            auto y = vector_of_four();

            SushiBLAS::BLASOps(first.get_recorder()).axpy(2.0, x, y);

            EXPECT_EQ(1u, first.take_nodes().size());
            EXPECT_TRUE(second.take_nodes().empty());
        }

        /** @brief Tests that a recorder returns the pool it was built on. */
        TEST_F(Unit_TaskRecorderTest, ARecorderHandsOutThePoolItWasGiven)
        {
            GraphProbe probe;

            EXPECT_EQ(&probe.get_recorder().get_scratch_pool(), &probe.get_scratch_pool());
            EXPECT_EQ(&engine->get_recorder().get_scratch_pool(), &engine->get_scratch_pool());
        }

        /**
         * @brief Tests that FORCE_REFERENCE refuses each family without a reference kernel, and
         *        AUTO records it.
         */
        TEST_F(Unit_TaskRecorderTest, ADeviceOnlyFamilyIsRefusedUnderForceReference)
        {
            GraphProbe probe(ExecutionMode::FORCE_REFERENCE);
            TaskRecorder& recorder = probe.get_recorder();

            auto a = vector_of_four();
            auto b = vector_of_four();
            auto c = vector_of_four();
            auto scalar = engine->create_tensor({1}, Core::DataType::FLOAT32);

            const std::array<DeviceOnlyCall, 5> calls = {{
                {"elementwise", [&] { SushiBLAS::ElementwiseOps(recorder).add(a, b, c); }},
                {"nonlinear", [&] { SushiBLAS::NonLinearOps(recorder).relu(c); }},
                {"logic", [&] { SushiBLAS::LogicOps(recorder).equal(a, b, c); }},
                {"reduction", [&] { SushiBLAS::ReductionOps(recorder).sum(a, scalar); }},
                {"random", [&] { SushiBLAS::RandomOps(recorder, engine->get_rng()).uniform(c); }},
            }};

            for (const DeviceOnlyCall& entry : calls)
            {
                SCOPED_TRACE(entry.family);
                EXPECT_THROW(entry.call(), std::invalid_argument);
            }
            EXPECT_TRUE(probe.take_nodes().empty());

            recorder.set_mode(ExecutionMode::AUTO);
            for (const DeviceOnlyCall& entry : calls)
            {
                SCOPED_TRACE(entry.family);
                EXPECT_NO_THROW(entry.call());
            }
            EXPECT_EQ(calls.size(), probe.take_nodes().size());
        }

        /** @brief Tests that a random call refused under FORCE_REFERENCE reserves no offset. */
        TEST_F(Unit_TaskRecorderTest, ARefusedRandomCallReservesNoOffset)
        {
            auto values = vector_of_four();
            auto matrix = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);

            engine->set_mode(ExecutionMode::FORCE_REFERENCE);
            const std::uint64_t offset_before = engine->get_rng().reserve(0).offset;

            EXPECT_THROW(SushiBLAS::RandomOps(*engine).uniform(values), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::RandomOps(*engine).shuffle(values), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::RandomOps(*engine).orthogonal(matrix), std::invalid_argument);

            EXPECT_EQ(offset_before, engine->get_rng().reserve(0).offset);
        }

        /** @brief Tests that a BLAS and a spatial operation record under FORCE_REFERENCE. */
        TEST_F(Unit_TaskRecorderTest, AKernelPairIsRecordedUnderForceReference)
        {
            GraphProbe probe(ExecutionMode::FORCE_REFERENCE);

            auto x = vector_of_four();
            auto y = vector_of_four();
            auto image = engine->create_tensor({1, 2, 2, 1}, Core::DataType::FLOAT32);
            auto columns = engine->create_tensor({4, 1}, Core::DataType::FLOAT32);
            const Spatial::ConvGeometry unit_window;

            EXPECT_NO_THROW(SushiBLAS::BLASOps(probe.get_recorder()).axpy(2.0, x, y));
            EXPECT_NO_THROW(
                SushiBLAS::SpatialOps(probe.get_recorder()).im2col(image, columns, unit_window));

            EXPECT_EQ(2u, probe.take_nodes().size());
        }

        /** @brief Tests that one chain of view calls gives two graphs equal edges and results. */
        TEST_F(Unit_TaskRecorderTest, TheSameCallsReplayIntoASecondGraph)
        {
            GraphProbe first;
            GraphProbe second;

            auto input = create_initialized_tensor<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
            ChainOutputs first_outputs = chain_outputs();
            ChainOutputs second_outputs = chain_outputs();

            record_chain(first.get_recorder(), input, first_outputs);
            record_chain(second.get_recorder(), input, second_outputs);

            const std::vector<int32_t> first_degrees = in_degrees_of(first.take_nodes());
            const std::vector<int32_t> second_degrees = in_degrees_of(second.take_nodes());
            EXPECT_EQ((std::vector<int32_t>{0, 1, 1}), first_degrees)
                << "mul has no producer; add reads mul's output and sum reads add's.";
            EXPECT_EQ(first_degrees, second_degrees);

            // take_nodes() emptied both graphs, so the chain is recorded again before it runs.
            record_chain(first.get_recorder(), input, first_outputs);
            record_chain(second.get_recorder(), input, second_outputs);
            first.execute();
            second.execute();

            const std::vector<float> expected_shifted = {2.0f, 6.0f, 12.0f, 20.0f};
            const std::vector<float> expected_total = {40.0f};
            EXPECT_TENSOR_NEAR(first_outputs.shifted, expected_shifted, 1e-5);
            EXPECT_TENSOR_NEAR(second_outputs.shifted, expected_shifted, 1e-5);
            EXPECT_TENSOR_NEAR(first_outputs.total, expected_total, 1e-5);
            EXPECT_TENSOR_NEAR(second_outputs.total, expected_total, 1e-5);
        }
    } // namespace Test
} // namespace SushiBLAS
