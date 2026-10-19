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
 * @file test_benchmark_harness.cpp
 * @brief Tests the size parsing, the names, the timing and the guard the benchmarks share.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
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

        /** @brief A benchmark body that throws, to exercise the guard. */
        struct ThrowingBody
        {
            /** @brief Throws a runtime_error. */
            void operator()(int, char**) const
            {
                throw std::runtime_error("refused");
            }
        };

        TEST(Medium_BenchmarkHarness, SizesComeFromTheArgumentsOrTheDefaults)
        {
            char program[] = "bench";
            char first[] = "64";
            char second[] = "1024";
            char bad[] = "12x";
            char* with_sizes[] = {program, first, second};
            char* without_sizes[] = {program};
            char* with_bad[] = {program, bad};

            EXPECT_EQ(Bench::parse_sizes(3, with_sizes, {1}), (std::vector<int64_t>{64, 1024}));
            EXPECT_EQ(Bench::parse_sizes(1, without_sizes, {7, 8}), (std::vector<int64_t>{7, 8}));
            EXPECT_THROW((void)Bench::parse_sizes(2, with_bad, {1}), std::invalid_argument);
        }

        TEST(Medium_BenchmarkHarness, NamesCoverTheModesTheLayoutsAndTheDtypes)
        {
            EXPECT_EQ(std::string(Bench::execution_mode_name(ExecutionMode::ForceSYCL)),
                      "ForceSYCL");
            EXPECT_EQ(std::string(Bench::layout_name(Core::Layout::RowMajor)), "RowMajor");
            EXPECT_EQ(std::string(Bench::layout_name(Core::Layout::ColumnMajor)), "ColumnMajor");
            EXPECT_EQ(std::string(Bench::dtype_name(Core::DataType::Float64)), "FLOAT64");
            EXPECT_EQ(Bench::dtype_bytes(Core::DataType::Float32), sizeof(float));
            EXPECT_THROW((void)Bench::dtype_name(Core::DataType::Int32), std::invalid_argument);
        }

        TEST(Medium_BenchmarkHarness, TimingRunsTheWarmUpAndTheTimedIterations)
        {
            int calls = 0;

            const Bench::Timing timing = Bench::time_iterations(CallCounter{&calls});

            EXPECT_EQ(calls, Bench::K_WARMUP_ITERATIONS + Bench::K_TIMED_ITERATIONS);
            EXPECT_LE(timing.min_seconds, timing.median_seconds);
        }

        TEST(Medium_BenchmarkHarness, TheGuardTurnsAThrowIntoAFailureCode)
        {
            char program[] = "bench";
            char* arguments[] = {program};

            EXPECT_EQ(Bench::run_guarded(ThrowingBody{}, 1, arguments), EXIT_FAILURE);
        }
    } // namespace Test
} // namespace SushiBLAS
