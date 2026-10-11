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
#include "test_utils.hpp"
#include "graph_probe.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the tolerance of a result made of small whole numbers: it is exact.
            constexpr double K_EXACT = 0.0;
        } // namespace

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

                /** @brief Names one record call of an operation family. */
                struct FamilyCall
                {
                    const char* family;         /**< The family, for the failure message. */
                    std::function<void()> call; /**< Records one operation of the family. */
                };

                /** @brief Returns a FLOAT32 vector of four elements from the fixture's engine. */
                [[nodiscard]] Tensor vector_of_four()
                {
                    return engine->create_tensor({4}, Core::DataType::Float32);
                }

                /** @brief Returns fresh outputs for one replay over a four-element input. */
                [[nodiscard]] ChainOutputs chain_outputs()
                {
                    return ChainOutputs{vector_of_four(), vector_of_four(),
                                        engine->create_tensor({1}, Core::DataType::Float32)};
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
        };

        /** @brief Tests that a record call adds its node to the recorder's graph only. */
        TEST_F(Unit_TaskRecorderTest, RecordsOnTheGraphItWasGiven)
        {
            GraphProbe first;
            GraphProbe second;

            auto x = vector_of_four();
            auto y = vector_of_four();

            SushiBLAS::BLASOps(first.recorder()).axpy(2.0, x, y);

            EXPECT_EQ(1u, first.take_recorded_count());
            EXPECT_EQ(0u, second.take_recorded_count());
        }

        /** @brief Tests that a recorder returns the pool it was built on. */
        TEST_F(Unit_TaskRecorderTest, ARecorderHandsOutThePoolItWasGiven)
        {
            GraphProbe probe;

            EXPECT_EQ(&probe.recorder().scratch_pool(), &probe.scratch_pool());
            EXPECT_EQ(&engine->recorder().scratch_pool(), &engine->scratch_pool());
        }

        /** @brief Tests that every family records and runs one task under ForceReference. */
        TEST_F(Unit_TaskRecorderTest, EveryFamilyRecordsUnderForceReference)
        {
            GraphProbe probe(ExecutionMode::ForceReference);
            TaskRecorder& recorder = probe.recorder();

            const std::vector<float> signed_values = {1.0f, -2.0f, 3.0f, -4.0f};
            const std::vector<float> counting_values = {1.0f, 2.0f, 3.0f, 4.0f};

            auto a = create_initialized_tensor<float>({4}, signed_values);
            auto b = create_initialized_tensor<float>({4}, counting_values);
            auto added = vector_of_four();
            auto rectified = create_initialized_tensor<float>({4}, signed_values);
            auto compared = vector_of_four();
            auto total = engine->create_tensor({1}, Core::DataType::Float32);
            auto drawn = vector_of_four();
            auto y = create_initialized_tensor<float>({4}, counting_values);
            auto image = create_initialized_tensor<float>({1, 2, 2, 1}, signed_values);
            auto columns = engine->create_tensor({4, 1}, Core::DataType::Float32);
            const Spatial::ConvGeometry unit_window;

            const std::array<FamilyCall, 7> calls = {{
                {"elementwise", [&] { SushiBLAS::ElementwiseOps(recorder).add(a, b, added); }},
                {"nonlinear", [&] { SushiBLAS::NonLinearOps(recorder).relu(rectified); }},
                {"logic", [&] { SushiBLAS::LogicOps(recorder).equal(a, b, compared); }},
                {"reduction", [&] { SushiBLAS::ReductionOps(recorder).sum(a, total); }},
                {"random",
                 [&] { SushiBLAS::RandomOps(recorder, engine->rng()).uniform(drawn); }},
                {"blas", [&] { SushiBLAS::BLASOps(recorder).axpy(2.0, a, y); }},
                {"spatial",
                 [&] { SushiBLAS::SpatialOps(recorder).im2col(image, columns, unit_window); }},
            }};

            for (const FamilyCall& entry : calls)
            {
                SCOPED_TRACE(entry.family);
                EXPECT_NO_THROW(entry.call());
                EXPECT_EQ(1u, probe.take_recorded_count());
            }

            // take_recorded_count() emptied the graph, so every call is recorded again before it runs.
            for (const FamilyCall& entry : calls)
                entry.call();
            probe.execute().value();

            EXPECT_TENSOR_NEAR(added, (std::vector<float>{2.0f, 0.0f, 6.0f, 0.0f}), K_EXACT);
            EXPECT_TENSOR_NEAR(rectified, (std::vector<float>{1.0f, 0.0f, 3.0f, 0.0f}), K_EXACT);
            EXPECT_TENSOR_NEAR(compared, (std::vector<float>{1.0f, 0.0f, 1.0f, 0.0f}), K_EXACT);
            EXPECT_TENSOR_NEAR(total, (std::vector<float>{-2.0f}), K_EXACT);
            EXPECT_TENSOR_NEAR(y, (std::vector<float>{3.0f, -2.0f, 9.0f, -4.0f}), K_EXACT);
            EXPECT_TENSOR_NEAR(columns, signed_values, K_EXACT);

            for (const float value : read_tensor<float>(drawn))
            {
                EXPECT_GE(value, 0.0f);
                EXPECT_LT(value, 1.0f);
            }
        }

        /** @brief Tests that one chain of view calls gives two graphs equal edges and results. */
        TEST_F(Unit_TaskRecorderTest, TheSameCallsReplayIntoASecondGraph)
        {
            GraphProbe first;
            GraphProbe second;

            auto input = create_initialized_tensor<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
            ChainOutputs first_outputs = chain_outputs();
            ChainOutputs second_outputs = chain_outputs();

            record_chain(first.recorder(), input, first_outputs);
            record_chain(second.recorder(), input, second_outputs);

            EXPECT_EQ(3u, first.recorded_count());
            EXPECT_EQ(3u, second.recorded_count());
            const auto first_counts = first.dependency_counts();
            const auto second_counts = second.dependency_counts();

            first.execute().value();
            second.execute().value();

            const std::vector<float> expected_shifted = {2.0f, 6.0f, 12.0f, 20.0f};
            const std::vector<float> expected_total = {40.0f};
            EXPECT_TENSOR_NEAR(first_outputs.shifted, expected_shifted, 1e-5);
            EXPECT_TENSOR_NEAR(second_outputs.shifted, expected_shifted, 1e-5);
            EXPECT_TENSOR_NEAR(first_outputs.total, expected_total, 1e-5);
            EXPECT_TENSOR_NEAR(second_outputs.total, expected_total, 1e-5);

            if (!first_counts || !second_counts)
                GTEST_SKIP() << GraphProbe::K_NO_RECORDED_TASK_VIEW;
            EXPECT_EQ((std::vector<std::size_t>{0, 1, 1}), *first_counts)
                << "mul has no producer; add reads mul's output and sum reads add's.";
            EXPECT_EQ(*first_counts, *second_counts);
        }

        /**
         * @brief Tests that a discarded task never runs and that the engine records and runs
         *        again afterwards.
         */
        TEST_F(Unit_TaskRecorderTest, ADiscardedTaskDoesNotRunAndTheEngineGoesOn)
        {
            auto x = create_initialized_tensor<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
            auto y = create_initialized_tensor<float>({4}, {10.0f, 20.0f, 30.0f, 40.0f});

            SushiBLAS::BLASOps(*engine).axpy(2.0, x, y);
            engine->discard_recorded();
            engine->execute().value();
            EXPECT_TENSOR_NEAR(y, (std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f}), 0.0);

            SushiBLAS::BLASOps(*engine).axpy(2.0, x, y);
            engine->execute().value();
            EXPECT_TENSOR_NEAR(y, (std::vector<float>{12.0f, 24.0f, 36.0f, 48.0f}), 1e-6);
        }

        /** @brief Tests that discarding an engine with nothing recorded does nothing. */
        TEST_F(Unit_TaskRecorderTest, DiscardingAnEmptyEngineIsHarmless)
        {
            auto y = create_initialized_tensor<float>({2}, {5.0f, 6.0f});

            engine->discard_recorded();
            engine->discard_recorded();
            engine->execute().value();
            EXPECT_TENSOR_NEAR(y, (std::vector<float>{5.0f, 6.0f}), 0.0);
        }
    } // namespace Test
} // namespace SushiBLAS
