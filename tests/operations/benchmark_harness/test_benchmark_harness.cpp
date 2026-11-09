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
#include <sstream>
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

        /** @brief Scripts the clock of time_iterations: reads alternate start and stop, and stop - start is the next duration. */
        struct ScriptedClock
        {
            const std::vector<double>* durations; /**< The duration of each timed iteration, in order. */
            std::size_t* reads;                   /**< Counts the reads made so far. */

            /** @brief Returns 0 for a start read and the iteration's duration for a stop read. */
            double operator()() const
            {
                const std::size_t read = (*reads)++;
                return read % 2 == 0 ? 0.0 : (*durations)[read / 2];
            }
        };

        /** @brief Ten durations in seconds whose sorted middle pair is 5 and 6, and whose shortest is 1. */
        const std::vector<double> K_SCRAMBLED_DURATIONS{10.0, 1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 5.0};

        /** @brief Times a counting body on the scripted clock and returns the timing. */
        Bench::Timing time_on_the_scripted_clock(int* calls)
        {
            std::size_t reads = 0;
            return Bench::time_iterations(CallCounter{calls}, ScriptedClock{&K_SCRAMBLED_DURATIONS, &reads});
        }

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

        TEST(Small_BenchmarkHarness, TimingRunsTheWarmUpAndTheTimedIterations)
        {
            int calls = 0;

            (void)time_on_the_scripted_clock(&calls);

            EXPECT_EQ(calls, Bench::K_WARMUP_ITERATIONS + Bench::K_TIMED_ITERATIONS);
        }

        TEST(Small_BenchmarkHarness, TimingReportsTheMeanOfTheMiddlePairForAnEvenCount)
        {
            int calls = 0;

            EXPECT_EQ(time_on_the_scripted_clock(&calls).median_seconds, 5.5);
        }

        TEST(Small_BenchmarkHarness, TimingReportsTheShortestIterationAsTheMinimum)
        {
            int calls = 0;

            EXPECT_EQ(time_on_the_scripted_clock(&calls).min_seconds, 1.0);
        }

        TEST(Small_BenchmarkHarness, TheGuardTurnsAThrowIntoAFailureCode)
        {
            char program[] = "bench";
            char* arguments[] = {program};
            std::string message;

            const int code = Bench::run_guarded(ThrowingBody{}, 1, arguments,
                                                [&message](const char* what) { message = what; });

            EXPECT_EQ(code, EXIT_FAILURE);
            EXPECT_EQ(message, "refused");
        }

        TEST(Small_BenchmarkHarness, TheTableHeaderNamesTheColumnsAndTheRateUnit)
        {
            std::ostringstream out;

            const Bench::ResultTable table("GB/s", out);

            EXPECT_EQ(out.str(),
                      "op         dtype            size    median ms       min ms       GB/s\n");
        }

        TEST(Small_BenchmarkHarness, APrintedRowFollowsTheMedianAndTheMinimumInMilliseconds)
        {
            std::ostringstream out;
            const Bench::ResultTable table("GB/s", out);
            out.str("");

            table.print_row("copy", Core::DataType::Float32, 64, Bench::Timing{0.002, 0.001}, 8.0, 2.0);

            EXPECT_EQ(out.str(),
                      "copy       FLOAT32            64        2.000        1.000    2000.00\n");
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
