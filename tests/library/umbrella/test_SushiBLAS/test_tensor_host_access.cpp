/**************************************************************************/
/* test_tensor_host_access.cpp                                            */
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
 * @file test_tensor_host_access.cpp
 * @brief Tests that a plan executed again reads the source the host rewrote through data_as.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/late_count_hold.hpp>
#include <SushiBLAS/engine/runtime_scratch_source.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds how many elements each tensor of the test has.
            constexpr std::int64_t K_ELEMENTS = 33;

            /// @brief Holds how many times the one compiled plan is executed.
            constexpr std::size_t K_EXECUTIONS = 3;

            /// @brief Holds the distance between the values of two executions, larger than K_ELEMENTS.
            constexpr std::int64_t K_EXECUTION_STEP = 100;

            /// @brief Sums what the executions of the plans on every device did.
            struct ReplayOutcome
            {
                std::size_t compiles = 0;    /**< Plans compiled, one per device. */
                std::size_t executions = 0;  /**< Executions that ran and succeeded. */
                std::size_t mismatches = 0;  /**< Destination elements that differ from the source. */
            };

            /// @brief Returns the value the host writes at element i before execution run.
            template<typename T>
            T source_value(std::size_t run, std::int64_t i)
            {
                return static_cast<T>(static_cast<std::int64_t>(run) * K_EXECUTION_STEP + i);
            }

            /// @brief Writes the values of execution run into source through its host address.
            template<typename T>
            void host_fill(Tensor& source, std::size_t run)
            {
                for (std::int64_t i = 0; i < source.num_elements; ++i)
                    source.data_as<T>()[i] = source_value<T>(run, i);
            }

            /// @brief Returns how many elements of destination differ from the source of execution run.
            template<typename T>
            std::size_t count_mismatches(const Tensor& destination, std::size_t run)
            {
                std::size_t mismatches = 0;
                for (std::int64_t i = 0; i < destination.num_elements; ++i)
                {
                    if (destination.data_as<T>()[i] != source_value<T>(run, i))
                        ++mismatches;
                }
                return mismatches;
            }

            /**
             * @brief Records Level1::copy, reached through BLASOps, once on one device, executes the plan K_EXECUTIONS times
             *        and rewrites the source on the host before each execution.
             */
            template<typename T>
            ReplayOutcome replay_copy_on(Engine& engine, std::size_t device)
            {
                SushiRuntime::Runtime& runtime = TestEnvironment::runtime();
                SushiRuntime::Graph graph = runtime.graph();
                RuntimeScratchSource scratch_source(runtime);
                ScratchPool scratch_pool(scratch_source);
                LateCountHold late_counts(runtime);
                TaskRecorder recorder(graph.recorder(), scratch_pool, late_counts);
                Tensor source = engine.create_tensor({K_ELEMENTS}, sushiblas_dtype<T>(),
                                                     SushiRuntime::Residency::Shared, device);
                Tensor destination = engine.create_tensor({K_ELEMENTS}, sushiblas_dtype<T>(),
                                                          SushiRuntime::Residency::Shared, device);
                BLASOps(recorder).copy(source, destination);

                ReplayOutcome outcome;
                for (std::size_t run = 0; run < K_EXECUTIONS; ++run)
                {
                    host_fill<T>(source, run);
                    SushiRuntime::RunReport report;
                    if (graph.run(report))
                        ++outcome.executions;
                    outcome.mismatches += count_mismatches<T>(destination, run);
                }
                outcome.compiles = graph.compile_count();
                graph.clear();
                late_counts.release();
                return outcome;
            }

            /// @brief Runs replay_copy_on on every device queue of the runtime and sums the outcomes.
            template<typename T>
            ReplayOutcome replay_copy_on_every_device(Engine& engine)
            {
                ReplayOutcome total;
                for (std::size_t device = 0; device < TestEnvironment::device_count(); ++device)
                {
                    const ReplayOutcome outcome = replay_copy_on<T>(engine, device);
                    total.compiles += outcome.compiles;
                    total.executions += outcome.executions;
                    total.mismatches += outcome.mismatches;
                }
                return total;
            }
        } // namespace

        /** @brief Builds tensors and runs plans on the fixture's engine. */
        class Medium_TensorHostAccess : public SushiBLASTest
        {

        };

        /** @brief Tests that each execution of one INT32 copy plan reads the source as the host last wrote it. */
        TEST_F(Medium_TensorHostAccess, AnInt32PlanExecutedThreeTimesCopiesTheRewrittenSource)
        {
            const ReplayOutcome outcome = replay_copy_on_every_device<std::int32_t>(*engine);

            EXPECT_EQ(outcome.compiles, TestEnvironment::device_count());
            EXPECT_EQ(outcome.executions, K_EXECUTIONS * TestEnvironment::device_count());
            EXPECT_EQ(outcome.mismatches, std::size_t{0});
        }

        /** @brief Tests that each execution of one FLOAT32 copy plan reads the source as the host last wrote it. */
        TEST_F(Medium_TensorHostAccess, AFloat32PlanExecutedThreeTimesCopiesTheRewrittenSource)
        {
            const ReplayOutcome outcome = replay_copy_on_every_device<float>(*engine);

            EXPECT_EQ(outcome.compiles, TestEnvironment::device_count());
            EXPECT_EQ(outcome.executions, K_EXECUTIONS * TestEnvironment::device_count());
            EXPECT_EQ(outcome.mismatches, std::size_t{0});
        }
    } // namespace Test
} // namespace SushiBLAS
