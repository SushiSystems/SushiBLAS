/**************************************************************************/
/* test_benchmark_graph.cpp                                               */
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
 * @file test_benchmark_graph.cpp
 * @brief Tests that a benchmark measures a task recorded on the graph it owns.
 * @author Mustafa Garip
 */

#include <atomic>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/benchmark_harness/benchmark_graph.hpp>
#include <SushiBLAS/benchmark_harness/benchmark_harness.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief A task body that counts its runs and orders no work. */
        struct CountingBody
        {
            std::atomic<int>* runs; /**< Receives one per run of the task. */

            /** @brief Adds one to the count and returns an event that is already complete. */
            sycl::event operator()(const KernelPairDispatcher&, sycl::queue&,
                                   const std::vector<sycl::event>&) const
            {
                runs->fetch_add(1);
                return sycl::event();
            }
        };

        TEST(Medium_BenchmarkGraph, MeasureRecordsAndRunsOneTaskPerIterationOnTheGraphItIsGiven)
        {
            SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
            Engine engine(runtime);
            Bench::BenchmarkGraph graph(engine);
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

            EXPECT_EQ(runs.load(), Bench::K_WARMUP_ITERATIONS + Bench::K_TIMED_ITERATIONS);
            EXPECT_LE(timing.min_seconds, timing.median_seconds);
            EXPECT_FALSE(graph.run().has_value());
        }
    } // namespace Test
} // namespace SushiBLAS
