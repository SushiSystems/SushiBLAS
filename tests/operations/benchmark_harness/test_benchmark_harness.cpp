/**************************************************************************/
/* test_benchmark_harness.cpp                                             */
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
 * @file test_benchmark_harness.cpp
 * @brief Tests the size parsing, the dtype names, the timing, measure and the guard the benchmarks share.
 * @author Mustafa Garip
 */

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/recording_graph/recording_graph.hpp>
#include <SushiBLAS/benchmark_harness/benchmark_harness.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Counts the calls it receives. */
        struct CallCounter
        {
            int* calls; /**< Receives one per call. */

            /** @brief Adds one to the count. */
            void operator()() const
            {
                ++*calls;
            }
        };

        /** @brief A benchmark body that returns at once. */
        struct NoOpBody
        {
            /** @brief Does nothing. */
            void operator()(int, char**) const
            {
            }
        };

        /** @brief A benchmark body that throws, to exercise the guard. */
        struct ThrowingBody
        {
            /** @brief Throws a runtime_error. */
            void operator()(int, char**) const
            {
                throw std::runtime_error("refused");
            }
        };

        /** @brief What one measure call over a counting task left behind. */
        struct MeasuredRun
        {
            Bench::Timing timing; /**< The timing measure returned. */
            int runs;             /**< How many times the recorded task ran. */
            bool graph_is_empty;  /**< Whether the graph held no task after measure. */
        };

        /** @brief A task body that counts its runs and orders no work. */
        struct CountingBody
        {
            std::atomic<int>* runs; /**< Receives one per run of the task. */

            /** @brief Adds one to the count and returns an event that is already complete. */
            sycl::event operator()(const KernelPairDispatcher&,
                                   const std::vector<sycl::event>&) const
            {
                runs->fetch_add(1);
                return sycl::event();
            }
        };

        /** @brief Measures a one-task recording on a fresh graph and reports what it left behind. */
        MeasuredRun measure_a_counting_task()
        {
            SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
            Engine engine(runtime);
            RecordingGraph graph(runtime, engine.scratch_pool(), engine.mode());
            Tensor out = engine.create_tensor({4});
            std::atomic<int> runs{0};

            const Bench::Timing timing = Bench::measure(graph, [&]()
            {
                OperandSet operands;
                operands.writes(out);
                TaskDescription description;
                description.name = "counted";
                graph.recorder().record_pair(description, operands, "test",
                                             CountingBody{&runs});
            });

            return MeasuredRun{timing, runs.load(), graph.is_empty()};
        }

        TEST(Small_BenchmarkHarness, ParseSizesReturnsTheArgumentsInOrder)
        {
            char program[] = "bench";
            char first[] = "64";
            char second[] = "1024";
            char* arguments[] = {program, first, second};

            EXPECT_EQ(Bench::parse_sizes(3, arguments, {1}), (std::vector<int64_t>{64, 1024}));
        }

        TEST(Small_BenchmarkHarness, ParseSizesReturnsTheDefaultsWhenNoSizeIsGiven)
        {
            char program[] = "bench";
            char* arguments[] = {program};

            EXPECT_EQ(Bench::parse_sizes(1, arguments, {7, 8}), (std::vector<int64_t>{7, 8}));
        }

        TEST(Small_BenchmarkHarness, ParseSizesRefusesAnArgumentThatIsNotANumber)
        {
            char program[] = "bench";
            char bad[] = "12x";
            char* arguments[] = {program, bad};

            EXPECT_THROW((void)Bench::parse_sizes(2, arguments, {1}), std::invalid_argument);
        }

        TEST(Small_BenchmarkHarness, ParseSizesRefusesZeroAndNegativeSizes)
        {
            char program[] = "bench";
            char zero[] = "0";
            char negative[] = "-3";
            char* with_zero[] = {program, zero};
            char* with_negative[] = {program, negative};

            EXPECT_THROW((void)Bench::parse_sizes(2, with_zero, {1}), std::invalid_argument);
            EXPECT_THROW((void)Bench::parse_sizes(2, with_negative, {1}), std::invalid_argument);
        }

        TEST(Small_BenchmarkHarness, DtypeNameRefusesADtypeTheBenchmarksDoNotRun)
        {
            EXPECT_THROW((void)Bench::dtype_name(Core::DataType::Int32), std::invalid_argument);
        }

        TEST(Small_BenchmarkHarness, TheGuardReturnsSuccessForABodyThatReturns)
        {
            char program[] = "bench";
            char* arguments[] = {program};

            EXPECT_EQ(Bench::run_guarded(NoOpBody{}, 1, arguments), EXIT_SUCCESS);
        }

        TEST(Medium_BenchmarkHarness, DeviceSupportFollowsTheAspectsOfTheDevice)
        {
            SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
            Engine engine(runtime);
            const sycl::device device = engine.create_tensor({1}).device();

            EXPECT_TRUE(Bench::device_supports(engine, Core::DataType::Float32));
            EXPECT_EQ(Bench::device_supports(engine, Core::DataType::Half),
                      device.has(sycl::aspect::fp16));
            EXPECT_EQ(Bench::device_supports(engine, Core::DataType::Float64),
                      device.has(sycl::aspect::fp64));
        }

        TEST(Medium_BenchmarkHarness, DeviceSupportOfComplex64FollowsTheFp64AspectOfTheDevice)
        {
            SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
            Engine engine(runtime);
            const sycl::device device = engine.create_tensor({1}).device();

            EXPECT_EQ(Bench::device_supports(engine, Core::DataType::Complex64),
                      device.has(sycl::aspect::fp64));
        }

        TEST(Medium_BenchmarkHarness, TimingRunsTheWarmUpAndTheTimedIterations)
        {
            int calls = 0;

            (void)Bench::time_iterations(CallCounter{&calls});

            EXPECT_EQ(calls, Bench::K_WARMUP_ITERATIONS + Bench::K_TIMED_ITERATIONS);
        }

        TEST(Medium_BenchmarkHarness, TimingReportsAMinimumNotAboveTheMedian)
        {
            int calls = 0;

            const Bench::Timing timing = Bench::time_iterations(CallCounter{&calls});

            EXPECT_LE(timing.min_seconds, timing.median_seconds);
        }

        TEST(Medium_BenchmarkHarness, TheGuardTurnsAThrowIntoAFailureCode)
        {
            char program[] = "bench";
            char* arguments[] = {program};

            EXPECT_EQ(Bench::run_guarded(ThrowingBody{}, 1, arguments), EXIT_FAILURE);
        }

        TEST(Medium_BenchmarkHarness, MeasureRunsOneTaskPerIteration)
        {
            const MeasuredRun measured = measure_a_counting_task();

            EXPECT_EQ(measured.runs, Bench::K_WARMUP_ITERATIONS + Bench::K_TIMED_ITERATIONS);
        }

        TEST(Medium_BenchmarkHarness, MeasureReportsAMinimumNotAboveTheMedian)
        {
            const MeasuredRun measured = measure_a_counting_task();

            EXPECT_LE(measured.timing.min_seconds, measured.timing.median_seconds);
        }

        TEST(Medium_BenchmarkHarness, MeasureLeavesTheGraphEmpty)
        {
            const MeasuredRun measured = measure_a_counting_task();

            EXPECT_TRUE(measured.graph_is_empty);
        }
    } // namespace Test
} // namespace SushiBLAS
