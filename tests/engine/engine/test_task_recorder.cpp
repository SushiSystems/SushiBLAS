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
#include <memory>
#include <thread>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <functional>
#include <optional>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the tolerance of a result made of small whole numbers: it is exact.
            constexpr double K_EXACT = 0.0;

            /// @brief Holds how many tasks each thread of the two-thread test records.
            constexpr std::size_t K_TASKS_PER_THREAD = 64;
        } // namespace

        /** @brief Builds the tensors and records the call chain the TaskRecorder tests share. */
        class Medium_TaskRecorderTest : public SushiBLASTest
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
        TEST_F(Medium_TaskRecorderTest, RecordsOnTheGraphItWasGiven)
        {
            GraphProbe first;
            GraphProbe second;

            auto x = vector_of_four();
            auto y = vector_of_four();

            SushiBLAS::BLASOps(first.recorder()).axpy(2.0, x, y);

            EXPECT_EQ(1u, first.take_recorded_count());
            EXPECT_EQ(0u, second.take_recorded_count());
        }

        /** @brief Tests that the determinism of a description reaches the runtime's plan. */
        TEST_F(Medium_TaskRecorderTest, ADescriptionsDeterminismReachesThePlanNode)
        {
            RecordingFixture fixture(*engine);
            Tensor out = vector_of_four();
            OperandSet operands;
            operands.writes(out);
            TaskDescription description;
            description.name = "stated";
            description.determinism = SushiRuntime::Determinism::Bitwise;

            fixture.recorder().record_pair(
                description, operands, "test",
                [](const KernelPairDispatcher&, sycl::queue&, const std::vector<sycl::event>&)
                {
                    return sycl::event();
                });

            SushiRuntime::Result<SushiRuntime::PlanView> plan = fixture.plan();
            ASSERT_TRUE(plan);
            ASSERT_EQ(plan->nodes.size(), std::size_t{1});
            EXPECT_EQ(plan->nodes[0].determinism, SushiRuntime::Determinism::Bitwise);
        }

        /** @brief Tests that a late count the runtime cannot read is refused before any task is recorded. */
        TEST_F(Medium_TaskRecorderTest, ALateCountOutsideHostInt32IsRefusedAndRecordsNothing)
        {
            for (const ExecutionMode mode :
                 {ExecutionMode::ForceReference, ExecutionMode::ForceSYCL})
            {
                RecordingFixture fixture(*engine);
                fixture.recorder().set_mode(mode);
                Tensor out = vector_of_four();
                OperandSet operands;
                operands.writes(out);
                const TaskDescription description;
                const Tensor shared_count = engine->create_tensor({1}, Core::DataType::Int32);

                EXPECT_THROW(fixture.recorder().record_late_pair(
                                 description, operands, "test", shared_count, 4,
                                 [] {}, [](std::size_t) {}),
                             std::invalid_argument);
                EXPECT_EQ(fixture.recorded_count(), std::size_t{0});
            }
        }

        /** @brief Tests that a late-count task of a live recorder runs the indices below its count. */
        TEST_F(Medium_TaskRecorderTest, ALateCountTaskRunsWhileItsRecorderLives)
        {
            SushiRuntime::Graph graph = engine->runtime().graph();
            ScratchPool scratch_pool(engine->runtime());
            LateCountHold late_counts(engine->runtime());
            TaskRecorder recorder(graph.recorder(), scratch_pool, late_counts);
            Tensor out = vector_of_four();
            Tensor count = engine->create_tensor({1}, Core::DataType::Int32,
                                                 SushiRuntime::Residency::Host);
            count.data_as<std::int32_t>()[0] = 2;
            for (std::int64_t i = 0; i < 4; ++i)
                out.data_as<float>()[i] = 0.0f;
            OperandSet operands;
            operands.writes(out);
            float* marks = out.data_as<float>();

            recorder.record_late_pair(
                TaskDescription(), operands, "test", count, 4, [] {},
                [marks](std::size_t i) { marks[i] = 1.0f; });
            SushiRuntime::RunReport report;
            const SushiRuntime::Result<SushiRuntime::RunReport*> ran = graph.run(report);

            ASSERT_TRUE(ran);
            EXPECT_EQ(out.data_as<float>()[0], 1.0f);
            EXPECT_EQ(out.data_as<float>()[1], 1.0f);
            EXPECT_EQ(out.data_as<float>()[2], 0.0f);
            EXPECT_EQ(out.data_as<float>()[3], 0.0f);
        }

        /**
         * @brief Tests that a run after the hold of a late-count task is gone is refused as
         *        DanglingResource, while the tensors are still alive and untouched.
         */
        TEST_F(Medium_TaskRecorderTest, ALateCountTaskRunAfterItsHoldIsGoneIsRefused)
        {
            SushiRuntime::Graph graph = engine->runtime().graph();
            ScratchPool scratch_pool(engine->runtime());
            Tensor out = vector_of_four();
            Tensor count = engine->create_tensor({1}, Core::DataType::Int32,
                                                 SushiRuntime::Residency::Host);
            count.data_as<std::int32_t>()[0] = 2;
            for (std::int64_t i = 0; i < 4; ++i)
                out.data_as<float>()[i] = 0.0f;
            OperandSet operands;
            operands.writes(out);
            float* marks = out.data_as<float>();

            {
                LateCountHold late_counts(engine->runtime());
                TaskRecorder recorder(graph.recorder(), scratch_pool, late_counts);
                recorder.record_late_pair(
                    TaskDescription(), operands, "test", count, 4, [] {},
                    [marks](std::size_t i) { marks[i] = 1.0f; });
            }
            SushiRuntime::RunReport report;
            const SushiRuntime::Result<SushiRuntime::RunReport*> ran = graph.run(report);

            ASSERT_FALSE(ran);
            EXPECT_EQ(ran.error().code, SushiRuntime::ErrorCode::DanglingResource);
            for (std::int64_t i = 0; i < 4; ++i)
                EXPECT_EQ(out.data_as<float>()[i], 0.0f);
        }

        /**
         * @brief Tests that a node recorded inside a runtime placement scope takes the scope's
         *        class and key, and a node recorded outside it takes neither.
         */
        TEST_F(Medium_TaskRecorderTest, ANodeInAPlacementScopeTakesItsClassAndKey)
        {
            using SushiRuntime::Literals::operator""_class;

            SushiRuntime::Runtime& runtime = engine->runtime();
            SushiRuntime::Graph graph = runtime.graph();
            ScratchPool scratch_pool(runtime);
            LateCountHold late_counts(runtime);
            TaskRecorder recorder(graph.recorder(), scratch_pool, late_counts);
            const SushiRuntime::PlacementScope scope =
                runtime.placement_scope("sushiai.replica", 3);

            auto x = vector_of_four();
            auto inside = vector_of_four();
            auto outside = vector_of_four();

            graph.recorder().within(scope, [&](SushiRuntime::Recorder&)
            {
                SushiBLAS::BLASOps(recorder).axpy(2.0, x, inside);
            });
            SushiBLAS::BLASOps(recorder).axpy(2.0, x, outside);

            SushiRuntime::Result<SushiRuntime::PlanView> plan = graph.plan();
            ASSERT_TRUE(plan);
            ASSERT_EQ(plan->nodes.size(), std::size_t{2});
            EXPECT_EQ(plan->nodes[0].placement_class, "sushiai.replica"_class);
            EXPECT_EQ(plan->nodes[0].placement_key, std::optional<std::uint64_t>{3});
            EXPECT_FALSE(plan->nodes[1].placement_class.is_named());
            EXPECT_EQ(plan->nodes[1].placement_key, std::nullopt);
        }

        /** @brief Tests that two threads recording on one recorder each get their own tasks. */
        TEST_F(Medium_TaskRecorderTest, TwoThreadsRecordOnOneRecorder)
        {
            GraphProbe probe;
            const Tensor first_x = vector_of_four();
            const Tensor second_x = vector_of_four();
            Tensor first_y = vector_of_four();
            Tensor second_y = vector_of_four();
            std::vector<TaskHandle> first_handles(K_TASKS_PER_THREAD);
            std::vector<TaskHandle> second_handles(K_TASKS_PER_THREAD);

            const auto record_all = [&probe](const Tensor& x, Tensor& y,
                                             std::vector<TaskHandle>& handles)
            {
                for (TaskHandle& handle : handles)
                    handle = SushiBLAS::BLASOps(probe.recorder()).axpy(2.0, x, y);
            };
            std::thread first([&] { record_all(first_x, first_y, first_handles); });
            std::thread second([&] { record_all(second_x, second_y, second_handles); });
            first.join();
            second.join();

            std::vector<TaskHandle> handles = first_handles;
            handles.insert(handles.end(), second_handles.begin(), second_handles.end());
            for (std::size_t i = 0; i < handles.size(); ++i)
            {
                EXPECT_TRUE(handles[i].is_valid());
                for (std::size_t j = i + 1; j < handles.size(); ++j)
                    EXPECT_TRUE(handles[i] != handles[j]);
            }
            EXPECT_EQ(2 * K_TASKS_PER_THREAD, probe.take_recorded_count());
        }

        /** @brief Tests that a recorder returns the pool it was built on. */
        TEST_F(Medium_TaskRecorderTest, ARecorderHandsOutThePoolItWasGiven)
        {
            GraphProbe probe;

            EXPECT_EQ(&probe.recorder().scratch_pool(), &probe.scratch_pool());
        }

        /** @brief Tests that every family records and runs one task under ForceReference. */
        TEST_F(Medium_TaskRecorderTest, EveryFamilyRecordsUnderForceReference)
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

            // take_recorded_count() emptied the graph, so every call is recorded again.
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
        TEST_F(Medium_TaskRecorderTest, TheSameCallsReplayIntoASecondGraph)
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
            ASSERT_TRUE(first_counts.has_value()) << first_counts.error().message;
            ASSERT_TRUE(second_counts.has_value()) << second_counts.error().message;

            first.execute().value();
            second.execute().value();

            const std::vector<float> expected_shifted = {2.0f, 6.0f, 12.0f, 20.0f};
            const std::vector<float> expected_total = {40.0f};
            EXPECT_TENSOR_NEAR(first_outputs.shifted, expected_shifted, 1e-5);
            EXPECT_TENSOR_NEAR(second_outputs.shifted, expected_shifted, 1e-5);
            EXPECT_TENSOR_NEAR(first_outputs.total, expected_total, 1e-5);
            EXPECT_TENSOR_NEAR(second_outputs.total, expected_total, 1e-5);

            EXPECT_EQ((std::vector<std::size_t>{0, 1, 1}), *first_counts)
                << "mul has no producer; add reads mul's output and sum reads add's.";
            EXPECT_EQ(*first_counts, *second_counts);
        }
    } // namespace Test
} // namespace SushiBLAS
