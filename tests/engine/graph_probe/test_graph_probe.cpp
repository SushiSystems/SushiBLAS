/**************************************************************************/
/* test_graph_probe.cpp                                                   */
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
 * @file test_graph_probe.cpp
 * @brief Tests that GraphProbe reports the tasks, the dependencies and the counts it recorded.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Records a task that does nothing over the given operands. */
        static void record_nothing(GraphProbe& probe, const char* name,
                                   const OperandSet& operands)
        {
            TaskDescription description;
            description.name = name;
            probe.recorder().record_pair(description, operands, "test",
                                         [](const KernelPairDispatcher& dispatcher,
                                            sycl::queue& q, const std::vector<sycl::event>& deps)
                                         {
                                             return dispatcher.run(
                                                 q, deps,
                                                 []()
                                                 {

                                                 },
                                                 [](sycl::queue&,
                                                    const std::vector<sycl::event>& events)
                                                 {
                                                     sycl::event::wait(events);
                                                     return sycl::event();
                                                 });
                                         });
        }

        /** @brief Records into probes over tensors of the fixture's engine. */
        class Unit_GraphProbe : public SushiBLASTest
        {

        };

        TEST_F(Unit_GraphProbe, AFreshProbeHasRecordedNothing)
        {
            GraphProbe probe;

            EXPECT_EQ(probe.recorded_count(), std::size_t{0});
            ASSERT_TRUE(probe.recorded_tasks().has_value());
            EXPECT_TRUE(probe.recorded_tasks()->empty());
        }

        TEST_F(Unit_GraphProbe, ReportsAReadAfterAWriteAsADependency)
        {
            const Tensor shared = engine->create_tensor({8});
            GraphProbe probe(ExecutionMode::ForceReference);
            record_nothing(probe, "producer", OperandSet().writes(shared));
            record_nothing(probe, "consumer", OperandSet().reads(shared));

            const auto counts = probe.dependency_counts();
            ASSERT_TRUE(counts.has_value());
            EXPECT_EQ(*counts, (std::vector<std::size_t>{0, 1}));
            const auto tasks = probe.recorded_tasks();
            ASSERT_TRUE(tasks.has_value());
            EXPECT_EQ((*tasks)[1].name, "consumer");
            EXPECT_EQ((*tasks)[1].reads.size(), std::size_t{1});
        }

        TEST_F(Unit_GraphProbe, TakeAndExecuteEmptyTheProbe)
        {
            GraphProbe probe(ExecutionMode::ForceReference);
            record_nothing(probe, "first", OperandSet());
            EXPECT_EQ(probe.take_recorded_count(), std::size_t{1});
            EXPECT_EQ(probe.recorded_count(), std::size_t{0});

            record_nothing(probe, "second", OperandSet());
            EXPECT_TRUE(probe.execute().has_value());
            EXPECT_EQ(probe.recorded_count(), std::size_t{0});
        }
    } // namespace Test
} // namespace SushiBLAS
