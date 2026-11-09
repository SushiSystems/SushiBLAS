/**************************************************************************/
/* test_task_dependencies.cpp                                             */
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
 * @file test_task_dependencies.cpp
 * @brief Tests that one recorded task of each recording seam runs and computes its result.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/random_kernels/random.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief The first input vector of the seams. */
            const std::vector<float> K_RAMP = {1.0f, 2.0f, 3.0f, 4.0f};

            /** @brief The second input vector of the seams. */
            const std::vector<float> K_TENS = {10.0f, 20.0f, 30.0f, 40.0f};
        } // namespace

        /** @brief Records one task on a probe, runs it and compares what it computed. */
        class Medium_TaskDependencies : public SushiBLASTest
        {
            protected:
                /** @brief Returns a FLOAT32 vector holding values. */
                [[nodiscard]] Tensor vector_of(const std::vector<float>& values)
                {
                    Tensor tensor = engine->create_tensor({static_cast<int64_t>(values.size())},
                                                          Core::DataType::Float32);
                    fill_tensor(tensor, values);
                    return tensor;
                }

                /**
                 * @brief Expects the one task record adds to compute what verify compares.
                 * @param record Callable `(TaskRecorder&) -> TaskHandle` that records one task.
                 * @param verify Callable `() -> void` comparing the task's output with the host
                 *        formula; it runs after the probe has run the task.
                 */
                template<typename Record, typename Verify>
                void expect_task_computes(const char* seam, Record&& record, Verify&& verify)
                {
                    SCOPED_TRACE(seam);

                    GraphProbe probe(ExecutionMode::ForceSYCL);
                    ASSERT_TRUE(record(probe.recorder()).is_valid());

                    ASSERT_EQ(1u, probe.recorded_count());
                    const auto tasks = probe.recorded_tasks();
                    ASSERT_TRUE(tasks.has_value()) << tasks.error().message;
                    ASSERT_EQ(1u, tasks->size());
                    EXPECT_EQ(0u, (*tasks)[0].dependency_count);

                    const SushiRuntime::Result<void> ran = probe.execute();
                    ASSERT_TRUE(ran.has_value()) << ran.error().message;
                    verify();
                }
        };

        /** @brief Tests that an elementwise add runs as a recorded task. */
        TEST_F(Medium_TaskDependencies, ARecordedElementwiseAddComputesItsResult)
        {
            auto a = vector_of(K_RAMP);
            auto b = vector_of(K_TENS);
            auto c = vector_of({0.0f, 0.0f, 0.0f, 0.0f});
            expect_task_computes("elementwise add",
                [&](TaskRecorder& r) { return SushiBLAS::ElementwiseOps(r).add(a, b, c); },
                [&] { expect_tensor_equal(c, std::vector<float>{11, 22, 33, 44}); });
        }

        /** @brief Tests that a nonlinear relu runs as a recorded task. */
        TEST_F(Medium_TaskDependencies, ARecordedReLUComputesItsResult)
        {
            auto t = vector_of({-1.0f, 2.0f, -3.0f, 4.0f});
            expect_task_computes("nonlinear relu",
                [&](TaskRecorder& r) { return SushiBLAS::NonLinearOps(r).relu(t); },
                [&] { expect_tensor_equal(t, std::vector<float>{0, 2, 0, 4}); });
        }

        /** @brief Tests that a logic equal runs as a recorded task. */
        TEST_F(Medium_TaskDependencies, ARecordedLogicEqualComputesItsResult)
        {
            auto a = vector_of(K_RAMP);
            auto b = vector_of({1.0f, 5.0f, 3.0f, 7.0f});
            auto same = vector_of({9.0f, 9.0f, 9.0f, 9.0f});
            expect_task_computes("logic equal",
                [&](TaskRecorder& r) { return SushiBLAS::LogicOps(r).equal(a, b, same); },
                [&] { expect_tensor_equal(same, std::vector<float>{1, 0, 1, 0}); });
        }

        /** @brief Tests that a reduction sum runs as a recorded task. */
        TEST_F(Medium_TaskDependencies, ARecordedSumComputesItsResult)
        {
            auto t = vector_of(K_RAMP);
            auto total = vector_of({0.0f});
            expect_task_computes("reduction sum",
                [&](TaskRecorder& r) { return SushiBLAS::ReductionOps(r).sum(t, total); },
                [&] { expect_tensor_equal(total, std::vector<float>{10}); });
        }

        /** @brief Tests that a uniform fill runs as a recorded task and draws the block of its offset. */
        TEST_F(Medium_TaskDependencies, ARecordedUniformFillComputesItsResult)
        {
            namespace Random = Kernels::SYCL::Random;

            constexpr std::uint64_t K_SEED = 20261007u;

            auto t = vector_of(std::vector<float>(Random::K_DRAWS_PER_BLOCK<float>, 9.0f));
            engine->rng().reseed(K_SEED);
            const std::uint64_t offset = engine->rng().reserve(0).offset;
            expect_task_computes("random uniform",
                [&](TaskRecorder& r)
                {
                    return SushiBLAS::RandomOps(r, engine->rng()).uniform(t, 0.0, 1.0);
                },
                [&]
                {
                    const auto draws = Random::uniform_block<float>(K_SEED, offset, 0, 0.0f, 1.0f);
                    const std::vector<float> expected(draws.begin(), draws.end());
                    EXPECT_TENSOR_NEAR(t, expected, absolute_tolerance<float>(2));
                });
        }

        /** @brief Tests that a BLAS axpy runs as a recorded task. */
        TEST_F(Medium_TaskDependencies, ARecordedAxpyComputesItsResult)
        {
            auto x = vector_of(K_RAMP);
            auto y = vector_of(K_TENS);
            expect_task_computes("blas axpy",
                [&](TaskRecorder& r) { return SushiBLAS::BLASOps(r).axpy(2.0, x, y); },
                [&] { expect_tensor_equal(y, std::vector<float>{12, 24, 36, 48}); });
        }

        /** @brief Tests that a spatial im2col runs as a recorded task. */
        TEST_F(Medium_TaskDependencies, ARecordedIm2colComputesItsResult)
        {
            // A 1x1 window gathers each pixel into its own row, so the columns equal the image.
            auto image = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);
            auto columns = engine->create_tensor({4, 1}, Core::DataType::Float32);
            fill_tensor(image, K_RAMP);
            fill_tensor(columns, {0.0f, 0.0f, 0.0f, 0.0f});
            const Spatial::ConvGeometry unit_window;
            expect_task_computes("spatial im2col",
                [&](TaskRecorder& r)
                {
                    return SushiBLAS::SpatialOps(r).im2col(image, columns, unit_window);
                },
                [&] { expect_tensor_equal(columns, K_RAMP); });
        }
    } // namespace Test
} // namespace SushiBLAS
