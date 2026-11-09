/**************************************************************************/
/* test_host_graph_probe.cpp                                              */
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
 * @file test_host_graph_probe.cpp
 * @brief Tests that HostGraphProbe counts the tasks each recorded task waits for from the
 *        regions the tasks declared.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor/host_block.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/graph_probe/host_graph_probe.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Records a task that does nothing over the given operands. */
        static void record_nothing(HostGraphProbe& probe, const char* name,
                                   const OperandSet& operands)
        {
            TaskDescription description;
            description.name = name;
            probe.recorder().record_pair(description, operands, "test",
                                         [](const KernelPairDispatcher& dispatcher,
                                            const std::vector<sycl::event>& deps)
                                         {
                                             return dispatcher.run(deps,
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

        /** @brief Records into probes over host tensors. */
        class Small_HostGraphProbe : public HostEngineTest
        {

        };

        TEST_F(Small_HostGraphProbe, AFreshProbeHasRecordedNothing)
        {
            HostGraphProbe probe;

            EXPECT_EQ(probe.recorded_count(), std::size_t{0});
            EXPECT_TRUE(probe.recorded_tasks().empty());
        }

        TEST_F(Small_HostGraphProbe, AReaderWaitsForTheWriterOfItsRegion)
        {
            const Tensor shared = engine->create_tensor({8});
            HostGraphProbe probe;
            record_nothing(probe, "producer", OperandSet().writes(shared));
            record_nothing(probe, "consumer", OperandSet().reads(shared));

            EXPECT_EQ(probe.dependency_counts(), (std::vector<std::size_t>{0, 1}));
            const std::vector<RecordedTask> tasks = probe.recorded_tasks();
            EXPECT_EQ(tasks[1].name, "consumer");
            EXPECT_EQ(tasks[1].reads.size(), std::size_t{1});
        }

        TEST_F(Small_HostGraphProbe, TwoTasksOnDisjointViewsOfOneStorageWaitForNothing)
        {
            constexpr int64_t K_COLUMNS = 4;
            auto storage = make_host_storage(static_cast<std::size_t>(2 * K_COLUMNS) * sizeof(float));
            const Tensor low(storage, {K_COLUMNS}, Core::DataType::Float32, 0);
            const Tensor high(storage, {K_COLUMNS}, Core::DataType::Float32, K_COLUMNS);
            HostGraphProbe probe;
            record_nothing(probe, "writes the low half", OperandSet().writes(low));
            record_nothing(probe, "reads the high half", OperandSet().reads(high));

            EXPECT_EQ(probe.dependency_counts(), (std::vector<std::size_t>{0, 0}));
        }

        TEST_F(Small_HostGraphProbe, AWriterWaitsForAnEarlierReaderAndAnEarlierWriter)
        {
            const Tensor shared = engine->create_tensor({8});
            HostGraphProbe probe;
            record_nothing(probe, "reader", OperandSet().reads(shared));
            record_nothing(probe, "first writer", OperandSet().writes(shared));
            record_nothing(probe, "second writer", OperandSet().writes(shared));

            EXPECT_EQ(probe.dependency_counts(), (std::vector<std::size_t>{0, 1, 2}));
        }

        TEST_F(Small_HostGraphProbe, AnEarlierTaskCountsOnceHoweverManyRegionsOverlap)
        {
            const Tensor first = engine->create_tensor({8});
            const Tensor second = engine->create_tensor({8});
            HostGraphProbe probe;
            record_nothing(probe, "producer", OperandSet().writes(first).writes(second));
            record_nothing(probe, "consumer", OperandSet().reads(first).reads(second));

            EXPECT_EQ(probe.dependency_counts(), (std::vector<std::size_t>{0, 1}));
        }
    } // namespace Test
} // namespace SushiBLAS
