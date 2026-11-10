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
 * @file test_task_recorder.cpp
 * @brief Tests TaskRecorder: the graph and pool it is bound to, and what each mode records.
 * @author Mustafa Garip
 */

#include <iterator>
#include <algorithm>
#include <set>
#include <atomic>
#include <string>
#include <memory>
#include <tuple>
#include <thread>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <functional>
#include <optional>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>
#include <SushiBLAS/graph_primitives/prefix_scan.hpp>
#include <SushiBLAS/graph_primitives/reduce.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>
#include <SushiBLAS/graph_primitives/combiners.hpp>
#include <SushiBLAS/cost/cost_table.hpp>
#include <SushiBLAS/profile/profile.hpp>
#if SB_ENABLE_PROFILER
#include <SushiProfiler/collector/collector.hpp>
#include <SushiProfiler/events/trace_event.hpp>
#include <SushiBLAS/engine/runtime_scratch_source.hpp>
#endif

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
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
            RecordingFixture fixture(TestEnvironment::runtime(), *engine);
            Tensor out = vector_of_four();
            OperandSet operands;
            operands.writes(out);
            TaskDescription description;
            description.name = "stated";
            description.determinism = SushiRuntime::Determinism::Bitwise;

            fixture.recorder().record_pair(
                description, operands, "test",
                [](const KernelPairDispatcher&, const std::vector<sycl::event>&)
                {
                    return sycl::event();
                });

            SushiRuntime::Result<SushiRuntime::PlanView> plan = fixture.plan();
            ASSERT_TRUE(plan);
            ASSERT_EQ(plan->nodes.size(), std::size_t{1});
            EXPECT_EQ(plan->nodes[0].determinism, SushiRuntime::Determinism::Bitwise);
        }

        namespace
        {
            /** @brief Names one execution mode. */
            struct ModeRow
            {
                const char* name;   /**< The row name a failure shows. */
                ExecutionMode mode; /**< The mode the recorder runs in. */
            };

            /** @brief Returns one row per mode that records a late-count pair. */
            std::vector<ModeRow> mode_rows()
            {
                return {{"ForceReference", ExecutionMode::ForceReference},
                        {"ForceSYCL", ExecutionMode::ForceSYCL}};
            }
        } // namespace

        /** @brief Runs one row of mode_rows(). */
        class Medium_TaskRecorderLateCountMode : public Medium_TaskRecorderTest,
                                                 public ::testing::WithParamInterface<ModeRow>
        {
        };

        /** @brief Tests that a late count the runtime cannot read is refused before any task is recorded. */
        TEST_P(Medium_TaskRecorderLateCountMode, ALateCountOutsideHostInt32IsRefusedAndRecordsNothing)
        {
            RecordingFixture fixture(TestEnvironment::runtime(), *engine);
            fixture.recorder().set_mode(GetParam().mode);
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

        INSTANTIATE_TEST_SUITE_P(
            , Medium_TaskRecorderLateCountMode, ::testing::ValuesIn(mode_rows()),
            [](const ::testing::TestParamInfo<ModeRow>& info)
            { return std::string(info.param.name); });

        /** @brief Tests that a late-count task of a live recorder runs the indices below its count. */
        TEST_F(Medium_TaskRecorderTest, ALateCountTaskRunsWhileItsRecorderLives)
        {
            SushiRuntime::Graph graph = TestEnvironment::runtime().graph();
            RuntimeScratchSource scratch_pool_source(TestEnvironment::runtime());
            ScratchPool scratch_pool(scratch_pool_source);
            LateCountHold late_counts(TestEnvironment::runtime());
            TaskRecorder recorder(graph.recorder(), scratch_pool, late_counts);
            Tensor out = create_initialized_tensor<float>({4}, {0.0f, 0.0f, 0.0f, 0.0f});
            Tensor count = engine->create_tensor({1}, Core::DataType::Int32,
                                                 SushiRuntime::Residency::Host);
            count.data_as<std::int32_t>()[0] = 2;
            OperandSet operands;
            operands.writes(out);
            float* marks = out.data_as<float>();

            recorder.record_late_pair(
                TaskDescription(), operands, "test", count, 4, [] {},
                [marks](std::size_t i) { marks[i] = 1.0f; });
            SushiRuntime::RunReport report;
            const SushiRuntime::Result<SushiRuntime::RunReport*> ran = graph.run(report);

            ASSERT_TRUE(ran);
            expect_tensor_equal(out, (std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f}));
        }

        /**
         * @brief Tests that a run after the hold of a late-count task is gone is refused as
         *        DanglingResource, while the tensors are still alive and untouched.
         */
        TEST_F(Medium_TaskRecorderTest, ALateCountTaskRunAfterItsHoldIsGoneIsRefused)
        {
            SushiRuntime::Graph graph = TestEnvironment::runtime().graph();
            RuntimeScratchSource scratch_pool_source(TestEnvironment::runtime());
            ScratchPool scratch_pool(scratch_pool_source);
            Tensor out = create_initialized_tensor<float>({4}, {0.0f, 0.0f, 0.0f, 0.0f});
            Tensor count = engine->create_tensor({1}, Core::DataType::Int32,
                                                 SushiRuntime::Residency::Host);
            count.data_as<std::int32_t>()[0] = 2;
            OperandSet operands;
            operands.writes(out);
            float* marks = out.data_as<float>();

            {
                LateCountHold late_counts(TestEnvironment::runtime());
                TaskRecorder recorder(graph.recorder(), scratch_pool, late_counts);
                recorder.record_late_pair(
                    TaskDescription(), operands, "test", count, 4, [] {},
                    [marks](std::size_t i) { marks[i] = 1.0f; });
            }
            SushiRuntime::RunReport report;
            const SushiRuntime::Result<SushiRuntime::RunReport*> ran = graph.run(report);

            ASSERT_FALSE(ran);
            EXPECT_EQ(ran.error().code, SushiRuntime::ErrorCode::DanglingResource);
            expect_tensor_equal(out, (std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f}));
        }

        /**
         * @brief Tests that a node recorded inside a runtime placement scope takes the scope's
         *        class and key, and a node recorded outside it takes neither.
         */
        TEST_F(Medium_TaskRecorderTest, ANodeInAPlacementScopeTakesItsClassAndKey)
        {
            using SushiRuntime::Literals::operator""_class;

            SushiRuntime::Runtime& runtime = TestEnvironment::runtime();
            SushiRuntime::Graph graph = runtime.graph();
            RuntimeScratchSource scratch_pool_source(runtime);
            ScratchPool scratch_pool(scratch_pool_source);
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

        /** @brief Holds one family call and the check of the tensors it wrote. */
        struct FamilyCase
        {
            std::function<void()> call;  /**< Records one operation of the family. */
            std::function<void()> check; /**< Compares the tensors the run wrote. */
        };

        class Medium_TaskRecorderFamily;

        /** @brief Names one operation family and builds its case. */
        struct FamilyRow
        {
            const char* name; /**< The row name a failure shows. */
            FamilyCase (*make)(Medium_TaskRecorderFamily&, TaskRecorder&); /**< Builds the case. */
        };

        /** @brief Runs one row of family_rows() and gives the rows the tensors of the fixture's engine. */
        class Medium_TaskRecorderFamily : public Medium_TaskRecorderTest,
                                          public ::testing::WithParamInterface<FamilyRow>
        {
            public:
                /** @brief Returns a FLOAT32 tensor of the given extents holding values. */
                [[nodiscard]] std::shared_ptr<Tensor>
                initialized(std::initializer_list<std::int64_t> extents,
                            const std::vector<float>& values)
                {
                    return std::make_shared<Tensor>(create_initialized_tensor<float>(extents, values));
                }

                /** @brief Returns an unwritten FLOAT32 tensor of the given extents. */
                [[nodiscard]] std::shared_ptr<Tensor> blank(std::initializer_list<std::int64_t> extents)
                {
                    return std::make_shared<Tensor>(
                        engine->create_tensor(extents, Core::DataType::Float32));
                }

                /** @brief Returns the random family's view over the engine's stream. */
                [[nodiscard]] SushiBLAS::RandomOps random_ops(TaskRecorder& recorder)
                {
                    return SushiBLAS::RandomOps(recorder, engine->rng());
                }
        };

        namespace
        {
            /** @brief Returns the values of the signed input every family row reads. */
            std::vector<float> signed_values()
            {
                return {1.0f, -2.0f, 3.0f, -4.0f};
            }

            /** @brief Returns the values of the counting input every family row reads. */
            std::vector<float> counting_values()
            {
                return {1.0f, 2.0f, 3.0f, 4.0f};
            }

            /** @brief Returns one row per operation family. */
            std::vector<FamilyRow> family_rows()
            {
                return {
                    {"Elementwise",
                     [](Medium_TaskRecorderFamily& f, TaskRecorder& recorder)
                     {
                         auto a = f.initialized({4}, signed_values());
                         auto b = f.initialized({4}, counting_values());
                         auto added = f.blank({4});
                         return FamilyCase{
                             [=, &recorder] { SushiBLAS::ElementwiseOps(recorder).add(*a, *b, *added); },
                             [=] { expect_tensor_equal(*added, (std::vector<float>{2.0f, 0.0f, 6.0f, 0.0f})); }};
                     }},
                    {"Nonlinear",
                     [](Medium_TaskRecorderFamily& f, TaskRecorder& recorder)
                     {
                         auto rectified = f.initialized({4}, signed_values());
                         return FamilyCase{
                             [=, &recorder] { SushiBLAS::NonLinearOps(recorder).relu(*rectified); },
                             [=] { expect_tensor_equal(*rectified, (std::vector<float>{1.0f, 0.0f, 3.0f, 0.0f})); }};
                     }},
                    {"Logic",
                     [](Medium_TaskRecorderFamily& f, TaskRecorder& recorder)
                     {
                         auto a = f.initialized({4}, signed_values());
                         auto b = f.initialized({4}, counting_values());
                         auto compared = f.blank({4});
                         return FamilyCase{
                             [=, &recorder] { SushiBLAS::LogicOps(recorder).equal(*a, *b, *compared); },
                             [=] { expect_tensor_equal(*compared, (std::vector<float>{1.0f, 0.0f, 1.0f, 0.0f})); }};
                     }},
                    {"Reduction",
                     [](Medium_TaskRecorderFamily& f, TaskRecorder& recorder)
                     {
                         auto a = f.initialized({4}, signed_values());
                         auto total = f.blank({1});
                         return FamilyCase{
                             [=, &recorder] { SushiBLAS::ReductionOps(recorder).sum(*a, *total); },
                             [=] { expect_tensor_equal(*total, (std::vector<float>{-2.0f})); }};
                     }},
                    {"Random",
                     [](Medium_TaskRecorderFamily& f, TaskRecorder& recorder)
                     {
                         auto drawn = f.blank({4});
                         return FamilyCase{
                             [=, &f, &recorder] { f.random_ops(recorder).uniform(*drawn); },
                             [=]
                             {
                                 const std::vector<float> values = read_tensor<float>(*drawn);
                                 EXPECT_TRUE(std::all_of(values.begin(), values.end(),
                                                         [](float value)
                                                         { return value >= 0.0f && value < 1.0f; }));
                             }};
                     }},
                    {"Blas",
                     [](Medium_TaskRecorderFamily& f, TaskRecorder& recorder)
                     {
                         auto a = f.initialized({4}, signed_values());
                         auto y = f.initialized({4}, counting_values());
                         return FamilyCase{
                             [=, &recorder] { SushiBLAS::BLASOps(recorder).axpy(2.0, *a, *y); },
                             [=] { expect_tensor_equal(*y, (std::vector<float>{3.0f, -2.0f, 9.0f, -4.0f})); }};
                     }},
                    {"Spatial",
                     [](Medium_TaskRecorderFamily& f, TaskRecorder& recorder)
                     {
                         auto image = f.initialized({1, 2, 2, 1}, signed_values());
                         auto columns = f.blank({4, 1});
                         return FamilyCase{
                             [=, &recorder]
                             {
                                 const Spatial::ConvGeometry unit_window;
                                 SushiBLAS::SpatialOps(recorder).im2col(*image, *columns, unit_window);
                             },
                             [=] { expect_tensor_equal(*columns, signed_values()); }};
                     }}};
            }
        } // namespace

        /** @brief Tests that a family records one task under ForceReference and runs it to the host result. */
        TEST_P(Medium_TaskRecorderFamily, TheFamilyRecordsOneTaskAndRunsItUnderForceReference)
        {
            GraphProbe probe(ExecutionMode::ForceReference);
            const FamilyCase family = GetParam().make(*this, probe.recorder());

            EXPECT_NO_THROW(family.call());
            EXPECT_EQ(1u, probe.recorded_count());
            probe.execute().value();

            family.check();
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_TaskRecorderFamily, ::testing::ValuesIn(family_rows()),
            [](const ::testing::TestParamInfo<FamilyRow>& info)
            { return std::string(info.param.name); });

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
            expect_tensor(first_outputs.shifted, expected_shifted, absolute_tolerance<float>(16));
            expect_tensor(second_outputs.shifted, expected_shifted, absolute_tolerance<float>(16));
            expect_tensor(first_outputs.total, expected_total, absolute_tolerance<float>(16));
            expect_tensor(second_outputs.total, expected_total, absolute_tolerance<float>(16));

            EXPECT_EQ((std::vector<std::size_t>{0, 1, 1}), *first_counts)
                << "mul has no producer; add reads mul's output and sum reads add's.";
            EXPECT_EQ(*first_counts, *second_counts);
        }

        /** @brief Records the calls whose cost the tests below read. */
        class Medium_TaskRecorderCost : public Medium_TaskRecorderTest
        {
            protected:
                /** @brief Returns the first task a probe's plan compiled to. */
                [[nodiscard]] static RecordedTask first_task_of(GraphProbe& probe)
                {
                    auto tasks = probe.recorded_tasks();
                    EXPECT_TRUE(tasks.has_value());
                    EXPECT_FALSE(tasks->empty());
                    return tasks.has_value() && !tasks->empty() ? tasks->front() : RecordedTask{};
                }

                /** @brief Returns a FLOAT32 tensor of the given extents. */
                [[nodiscard]] Tensor floats(std::initializer_list<std::int64_t> extents)
                {
                    return engine->create_tensor(extents, Core::DataType::Float32);
                }
        };

        /** @brief Records on a host engine and reads the cost each task stated back from the host sink. */
        class Small_TaskRecorderCost : public HostEngineTest
        {
            protected:
                /** @brief Returns a FLOAT32 host tensor of the given extents. */
                [[nodiscard]] Tensor floats(std::initializer_list<std::int64_t> extents)
                {
                    return engine->create_tensor(extents, Core::DataType::Float32);
                }

                /** @brief Records a 2x3 by 3x4 GEMM into a 2x4 result with the given beta. */
                void record_gemm(double beta)
                {
                    const Tensor a = floats({2, 3});
                    const Tensor b = floats({3, 4});
                    Tensor c = floats({2, 4});
                    SushiBLAS::BLASOps(recorder()).gemm(a, b, c, Core::Transpose::NoTrans,
                                                        Core::Transpose::NoTrans, 1.0, beta);
                }

                /** @brief Returns the one task the test recorded. */
                [[nodiscard]] HostTaskRecord only_record() const
                {
                    const std::vector<HostTaskRecord> records = recording->records();
                    EXPECT_EQ(records.size(), std::size_t{1});
                    return records.empty() ? HostTaskRecord{} : records.front();
                }
        };

        /** @brief Tests that a GEMM node states 2*m*n*k FLOPs. */
        TEST_F(Small_TaskRecorderCost, AGemmNodeStatesTwoMNKFlops)
        {
            record_gemm(0.0);

            const Cost cost = only_record().cost;
            EXPECT_TRUE(cost.is_costed());
            EXPECT_EQ(cost.flops, std::uint64_t{2 * 2 * 3 * 4});
        }

        /** @brief Tests that a GEMM node states the bytes of A, B and a C that is written only. */
        TEST_F(Small_TaskRecorderCost, AGemmNodeStatesTheBytesOfItsOperands)
        {
            record_gemm(0.0);

            EXPECT_EQ(only_record().cost.bytes, std::uint64_t{(6 + 12 + 8) * sizeof(float)});
        }

        /** @brief Tests that a GEMM node of FLOAT32 operands states the FLOAT32 class. */
        TEST_F(Small_TaskRecorderCost, AGemmNodeStatesTheFloat32Class)
        {
            record_gemm(0.0);

            EXPECT_EQ(only_record().cost.dtype, Core::DataType::Float32);
        }

        /** @brief Tests that a beta of zero declares C as written only: two reads and one write. */
        TEST_F(Small_TaskRecorderCost, ABetaZeroGemmReadsNoC)
        {
            record_gemm(0.0);

            const HostTaskRecord record = only_record();
            EXPECT_EQ(record.reads.size(), std::size_t{2});
            EXPECT_EQ(record.writes.size(), std::size_t{1});
        }

        /** @brief Tests that a beta of one updates C, so its bytes count twice. */
        TEST_F(Small_TaskRecorderCost, AnUpdatedOperandCountsTwiceInTheBytes)
        {
            record_gemm(1.0);

            EXPECT_EQ(only_record().cost.bytes, std::uint64_t{(6 + 12 + 2 * 8) * sizeof(float)});
        }

        /** @brief Tests that an elementwise node states one FLOP per element. */
        TEST_F(Small_TaskRecorderCost, AnElementwiseNodeStatesItsElementCount)
        {
            const Tensor a = floats({4});
            const Tensor b = floats({4});
            Tensor out = floats({4});

            SushiBLAS::ElementwiseOps(recorder()).add(a, b, out);

            const Cost cost = only_record().cost;
            EXPECT_EQ(cost.flops, std::uint64_t{4});
            EXPECT_EQ(cost.bytes, std::uint64_t{3 * 4 * sizeof(float)});
        }

        /** @brief Tests that a task whose call site states no shape and whose body returns an empty event has no cost. */
        TEST_F(Small_TaskRecorderCost, AnEmptyEventTaskIsUncosted)
        {
            Tensor out = floats({4});
            OperandSet operands;
            operands.writes(out);

            recorder().record_pair(
                TaskDescription(), operands, "test",
                [](const KernelPairDispatcher&, const std::vector<sycl::event>&)
                {
                    return sycl::event();
                });

            EXPECT_FALSE(only_record().cost.is_costed());
        }

        /** @brief Tests that a late-count task, whose width is read at run time, states no cost. */
        TEST_F(Medium_TaskRecorderCost, ALateCountTaskIsUncosted)
        {
            GraphProbe probe;
            Tensor out = floats({4});
            Tensor count = engine->create_tensor({1}, Core::DataType::Int32,
                                                 SushiRuntime::Residency::Host);
            count.data_as<std::int32_t>()[0] = 2;
            OperandSet operands;
            operands.writes(out);
            TaskDescription description;
            description.operation = SushiRuntime::OperationID(SushiRuntime::OperationID::hash("math.ew.add", 11));
            description.shape = ElementwiseShape{4, Core::DataType::Float32};

            probe.recorder().record_late_pair(description, operands, "test", count, 4, [] {},
                                              [](std::size_t) {});

            EXPECT_FALSE(first_task_of(probe).cost.has_value());
        }

        /** @brief Tests that a node keeps its cost when the plan is rebuilt after another record call. */
        TEST_F(Medium_TaskRecorderCost, TheCostSurvivesAPlanRebuild)
        {
            GraphProbe probe;
            const Tensor a = floats({2, 3});
            const Tensor b = floats({3, 4});
            Tensor c = floats({2, 4});
            SushiBLAS::BLASOps(probe.recorder()).gemm(a, b, c);
            const RecordedTask before = first_task_of(probe);

            Tensor d = floats({2, 4});
            SushiBLAS::ElementwiseOps(probe.recorder()).add(c, c, d);
            auto tasks = probe.recorded_tasks();

            ASSERT_TRUE(tasks.has_value());
            ASSERT_EQ(tasks->size(), std::size_t{2});
            ASSERT_TRUE(before.cost.has_value());
            ASSERT_TRUE((*tasks)[0].cost.has_value());
            EXPECT_EQ((*tasks)[0].cost->flops, before.cost->flops);
            EXPECT_EQ((*tasks)[0].cost->bytes, before.cost->bytes);
        }

        /** @brief Holds the checks of the join between a record call and a runtime rule. */
        class Medium_TaskRecorderJoin : public Medium_TaskRecorderTest
        {
        };

        /**
         * @brief Tests that each record call adds exactly one rule, at the node count the graph
         *        had before it, and that the runtime maps every rule to its own registration index.
         */
        TEST_F(Medium_TaskRecorderJoin, EachRecordCallAddsOneRuleWithItsOwnRegistrationIndex)
        {
            SushiRuntime::Graph graph = TestEnvironment::runtime().graph();
            RuntimeScratchSource scratch_pool_source(TestEnvironment::runtime());
            ScratchPool scratch_pool(scratch_pool_source);
            LateCountHold late_counts(TestEnvironment::runtime());
            TaskRecorder recorder(graph.recorder(), scratch_pool, late_counts);
            const Tensor a = engine->create_tensor({2, 3}, Core::DataType::Float32);
            const Tensor b = engine->create_tensor({3, 4}, Core::DataType::Float32);
            Tensor c = engine->create_tensor({2, 4}, Core::DataType::Float32);
            Tensor d = engine->create_tensor({2, 4}, Core::DataType::Float32);

            std::size_t expected = graph.recorder().node_count();
            SushiBLAS::BLASOps(recorder).gemm(a, b, c);
            EXPECT_EQ(graph.recorder().node_count(), ++expected);
            SushiBLAS::ElementwiseOps(recorder).add(c, c, d);
            EXPECT_EQ(graph.recorder().node_count(), ++expected);
            SushiBLAS::NonLinearOps(recorder).relu(d);
            EXPECT_EQ(graph.recorder().node_count(), ++expected);

            SushiRuntime::Result<SushiRuntime::PlanView> plan = graph.plan();
            ASSERT_TRUE(plan);
            ASSERT_EQ(plan->nodes.size(), std::size_t{3});
            std::vector<std::uint32_t> indices;
            std::transform(plan->nodes.begin(), plan->nodes.end(), std::back_inserter(indices),
                           [&graph](const SushiRuntime::PlanNode& node)
                           { return graph.rule_registration(node.rule); });
            const std::set<std::uint32_t> registrations(indices.begin(), indices.end());
            EXPECT_EQ(registrations.size(), std::size_t{3});
            EXPECT_EQ(registrations.count(SushiRuntime::Graph::K_NO_REGISTRATION), std::size_t{0});
        }

        /** @brief Tests that a late-count pair is one rule. */
        TEST_F(Medium_TaskRecorderJoin, ALateCountPairIsOneRule)
        {
            SushiRuntime::Graph graph = TestEnvironment::runtime().graph();
            RuntimeScratchSource scratch_pool_source(TestEnvironment::runtime());
            ScratchPool scratch_pool(scratch_pool_source);
            LateCountHold late_counts(TestEnvironment::runtime());
            TaskRecorder recorder(graph.recorder(), scratch_pool, late_counts);
            Tensor out = engine->create_tensor({4}, Core::DataType::Float32);
            Tensor count = engine->create_tensor({1}, Core::DataType::Int32,
                                                 SushiRuntime::Residency::Host);
            OperandSet operands;
            operands.writes(out);
            const std::size_t before = graph.recorder().node_count();

            recorder.record_late_pair(TaskDescription(), operands, "test", count, 4, [] {},
                                      [](std::size_t) {});

            EXPECT_EQ(graph.recorder().node_count(), before + 1);
        }

        /** @brief Records on a host engine and reads the origin label each task carried from the host sink. */
        class Small_TaskRecorderOrigin : public HostEngineTest
        {
            protected:
                /** @brief Records one add of two fresh FLOAT32 vectors of four elements. */
                void record_add()
                {
                    const Tensor a = engine->create_tensor({4}, Core::DataType::Float32);
                    Tensor out = engine->create_tensor({4}, Core::DataType::Float32);
                    SushiBLAS::ElementwiseOps(recorder()).add(a, a, out);
                }

                /** @brief Returns the origin of every recorded task, in record order. */
                [[nodiscard]] std::vector<std::string> origins() const
                {
                    std::vector<std::string> labels;
                    for (const HostTaskRecord& record : recording->records())
                        labels.push_back(record.origin);
                    return labels;
                }
        };

        /** @brief Tests that tasks inside a scope carry its label, and a task before it and one after it carry none. */
        TEST_F(Small_TaskRecorderOrigin, TasksInAScopeCarryItsLabelAndTheOnesAroundItCarryNone)
        {
            record_add();
            {
                const TaskRecorder::OriginScope scope = recorder().origin("layer.3");
                record_add();
                record_add();
            }
            record_add();

            EXPECT_EQ((std::vector<std::string>{"", "layer.3", "layer.3", ""}), origins());
        }

        /** @brief Tests that a nested scope restores the label of the scope around it. */
        TEST_F(Small_TaskRecorderOrigin, ANestedScopeRestoresTheOuterLabel)
        {
            {
                const TaskRecorder::OriginScope outer = recorder().origin("outer");
                {
                    const TaskRecorder::OriginScope inner = recorder().origin("inner");
                    record_add();
                }
                record_add();
            }

            EXPECT_EQ((std::vector<std::string>{"inner", "outer"}), origins());
        }

        /** @brief Tests that a moved scope restores the label once, when the last owner ends. */
        TEST_F(Small_TaskRecorderOrigin, AMovedScopeRestoresTheLabelOnce)
        {
            {
                TaskRecorder::OriginScope moved_from = recorder().origin("moved");
                const TaskRecorder::OriginScope moved_to = std::move(moved_from);
                record_add();
            }
            record_add();

            EXPECT_EQ((std::vector<std::string>{"moved", ""}), origins());
        }

#if SB_ENABLE_PROFILER
        /** @brief Holds the checks of the zones a record call writes into an open collector. */
        class Medium_TaskRecorderZones : public Medium_TaskRecorderTest
        {
            protected:
                void SetUp() override
                {
                    Medium_TaskRecorderTest::SetUp();
                    empty_the_collector();
                    ASSERT_TRUE(SushiProfiler::Collector::instance().open(SushiProfiler::CollectorSettings{}));
                }

                void TearDown() override
                {
                    empty_the_collector();
                    Medium_TaskRecorderTest::TearDown();
                }

                /** @brief Returns how many events of one kind the calling thread wrote since the last drain. */
                static std::size_t events_of(SushiProfiler::EventKind kind)
                {
                    SushiProfiler::DrainBatch batch;
                    SushiProfiler::Collector::instance().drain(batch);
                    std::size_t found = 0;
                    for (const SushiProfiler::ThreadDrain& thread : batch.threads)
                    {
                        if (thread.thread_index != SushiProfiler::Collector::thread_index())
                            continue;
                        for (const SushiProfiler::TraceEvent& event : thread.events)
                            found += SushiProfiler::kind_of(event) == kind ? 1U : 0U;
                    }
                    return found;
                }

            private:
                /** @brief Closes the collector and discards what its rings hold. */
                static void empty_the_collector()
                {
                    SushiProfiler::Collector::instance().close();
                    SushiProfiler::DrainBatch discarded;
                    SushiProfiler::Collector::instance().drain(discarded);
                }
        };

        /** @brief Tests that a record call writes a zone while a collector is open and none once it is closed. */
        TEST_F(Medium_TaskRecorderZones, ARecordCallWritesAZoneOnlyWhileTheCollectorIsOpen)
        {
            GraphProbe probe;
            const Tensor a = vector_of_four();
            Tensor out = vector_of_four();

            SushiBLAS::ElementwiseOps(probe.recorder()).add(a, a, out);
            const std::size_t while_open = events_of(SushiProfiler::EventKind::Zone);
            SushiProfiler::Collector::instance().close();
            SushiBLAS::ElementwiseOps(probe.recorder()).add(a, a, out);
            const std::size_t while_closed = events_of(SushiProfiler::EventKind::Zone);

            EXPECT_GE(while_open, std::size_t{1});
            EXPECT_EQ(while_closed, std::size_t{0});
        }
#endif

        namespace
        {
            /** @brief Names one operation and the call that records it alone. */
            struct CostCoverageRow
            {
                const char* name;                       /**< The row name a failure shows. */
                void (*record)(Engine&, TaskRecorder&); /**< Records the one operation of the row. */
            };

            /** @brief Returns a FLOAT32 tensor of the given extents. */
            Tensor floats_of(Engine& engine, std::initializer_list<std::int64_t> extents)
            {
                return engine.create_tensor(extents, Core::DataType::Float32);
            }

            /** @brief Returns one row per operation of an inference-like graph. */
            std::vector<CostCoverageRow> cost_coverage_rows()
            {
                return {
                    {"WindowDot",
                     [](Engine& engine, TaskRecorder& recorder)
                     {
                         Spatial::ConvGeometry window;
                         window.kernel_h = 3;
                         window.kernel_w = 3;
                         const Tensor image = floats_of(engine, {1, 4, 4, 2});
                         const Tensor taps = floats_of(engine, {3, 3, 2});
                         Tensor features = floats_of(engine, {1, 2, 2, 2});
                         SushiBLAS::SpatialOps(recorder).window_dot(image, taps, features, window);
                     }},
                    {"Gemm",
                     [](Engine& engine, TaskRecorder& recorder)
                     {
                         const Tensor left = floats_of(engine, {8, 8});
                         const Tensor right = floats_of(engine, {8, 8});
                         Tensor product = floats_of(engine, {8, 8});
                         SushiBLAS::BLASOps(recorder).gemm(left, right, product);
                     }},
                    {"Relu",
                     [](Engine& engine, TaskRecorder& recorder)
                     {
                         Tensor product = floats_of(engine, {8, 8});
                         SushiBLAS::NonLinearOps(recorder).relu(product);
                     }},
                    {"ConcatChannels",
                     [](Engine& engine, TaskRecorder& recorder)
                     {
                         const Tensor features = floats_of(engine, {1, 2, 2, 2});
                         Tensor joined = floats_of(engine, {1, 2, 2, 4});
                         SushiBLAS::SpatialOps(recorder).concat_channels(features, joined, 0);
                     }},
                    {"UpsampleNearest",
                     [](Engine& engine, TaskRecorder& recorder)
                     {
                         const Tensor features = floats_of(engine, {1, 2, 2, 2});
                         Tensor enlarged = floats_of(engine, {1, 4, 4, 2});
                         SushiBLAS::SpatialOps(recorder).upsample_nearest(features, enlarged, 2, 2);
                     }},
                    {"Sum",
                     [](Engine& engine, TaskRecorder& recorder)
                     {
                         const Tensor product = floats_of(engine, {8, 8});
                         Tensor total = floats_of(engine, {1});
                         SushiBLAS::ReductionOps(recorder).sum(product, total);
                     }}};
            }
        } // namespace

        /** @brief Runs one row of cost_coverage_rows(). */
        class Medium_TaskRecorderCostCoverage : public Medium_TaskRecorderTest,
                                                public ::testing::WithParamInterface<CostCoverageRow>
        {
        };

        /** @brief Tests that the node of an operation of an inference-like graph is costed or has a row. */
        TEST_P(Medium_TaskRecorderCostCoverage, TheNodeStatesACostOrHasARowInTheCostTable)
        {
            GraphProbe probe;

            GetParam().record(*engine, probe.recorder());

            auto tasks = probe.recorded_tasks();
            ASSERT_TRUE(tasks.has_value());
            ASSERT_EQ(tasks->size(), std::size_t{1});
            const RecordedTask& task = tasks->front();
            EXPECT_TRUE(task.operation.is_named()) << task.name;
            EXPECT_TRUE(task.cost.has_value() || find_cost_row(task.operation) != nullptr)
                << task.name << " states no cost and has no row in the cost table";
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_TaskRecorderCostCoverage, ::testing::ValuesIn(cost_coverage_rows()),
            [](const ::testing::TestParamInfo<CostCoverageRow>& info)
            { return std::string(info.param.name); });

        namespace
        {
            /** @brief Names one graph primitive and the call that records it. */
            struct PrimitiveRow
            {
                const char* name; /**< The row name a failure shows. */
                /**
                 * Records the primitive and returns the buffers it names: the graph refuses a plan
                 * over a freed allocation, so the caller keeps them until it has read the plan.
                 */
                std::shared_ptr<void> (*record)(SushiRuntime::Runtime&,
                                                GraphPrimitives::RuntimePrimitiveRecorder&);
            };

            /** @brief Returns one row per graph primitive. */
            std::vector<PrimitiveRow> primitive_rows()
            {
                return {
                    {"PrefixScan",
                     [](SushiRuntime::Runtime& runtime,
                        GraphPrimitives::RuntimePrimitiveRecorder& port)
                     {
                         auto counts = runtime.buffer<std::uint32_t>(64).value();
                         auto offsets = runtime.buffer<std::uint32_t>(65).value();
                         auto total = runtime.buffer<std::uint32_t>(1).value();
                         GraphPrimitives::prefix_scan(port, counts, offsets, total,
                                                      SushiRuntime::LateCount::fixed(64))
                             .value();
                         return std::shared_ptr<void>(std::make_shared<
                             std::tuple<decltype(counts), decltype(offsets), decltype(total)>>(
                             std::move(counts), std::move(offsets), std::move(total)));
                     }},
                    {"Reduce",
                     [](SushiRuntime::Runtime& runtime,
                        GraphPrimitives::RuntimePrimitiveRecorder& port)
                     {
                         auto values = runtime.buffer<float>(64).value();
                         auto sum = runtime.buffer<float>(1).value();
                         GraphPrimitives::reduce(port, values, sum, std::size_t{64},
                                                 GraphPrimitives::Sum<float>{})
                             .value();
                         return std::shared_ptr<void>(
                             std::make_shared<std::tuple<decltype(values), decltype(sum)>>(
                                 std::move(values), std::move(sum)));
                     }}};
            }
        } // namespace

        /** @brief Runs one row of primitive_rows(). */
        class Medium_TaskRecorderPrimitiveCoverage
            : public Medium_TaskRecorderTest,
              public ::testing::WithParamInterface<PrimitiveRow>
        {
        };

        /** @brief Tests that every node a graph primitive records carries an operation id. */
        TEST_P(Medium_TaskRecorderPrimitiveCoverage, EveryNodeCarriesAnOperationId)
        {
            SushiRuntime::Runtime& runtime = TestEnvironment::runtime();
            auto graph = runtime.graph();
            GraphPrimitives::RuntimePrimitiveRecorder port(graph.recorder());

            const std::shared_ptr<void> buffers = GetParam().record(runtime, port);

            const SushiRuntime::Result<SushiRuntime::PlanView> plan = graph.plan();
            ASSERT_TRUE(plan);
            ASSERT_FALSE(plan->nodes.empty());
            EXPECT_TRUE(std::all_of(plan->nodes.begin(), plan->nodes.end(),
                                    [](const SushiRuntime::PlanNode& node)
                                    { return node.operation.is_named(); }));
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_TaskRecorderPrimitiveCoverage, ::testing::ValuesIn(primitive_rows()),
            [](const ::testing::TestParamInfo<PrimitiveRow>& info)
            { return std::string(info.param.name); });
    } // namespace Test
} // namespace SushiBLAS
