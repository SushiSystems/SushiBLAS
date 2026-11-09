/**************************************************************************/
/* recording_graph_tasks.hpp                                              */
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
 * @file recording_graph_tasks.hpp
 * @brief Declares the two tasks the RecordingGraph tests record: one that does nothing and one with a late count.
 * @author Mustafa Garip
 * @see tests/engine/recording_graph/README.md
 */

#pragma once

#include <cstddef>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/recording_graph/recording_graph.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Records a task that does nothing and reads or writes nothing. */
        inline void record_nothing(RecordingGraph& graph)
        {
            TaskDescription description;
            description.name = "nothing";
            graph.recorder().record_pair(description, OperandSet(), "test",
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

        /** @brief Records one late-count task over out, whose width count holds, on graph. */
        inline void record_late_task(RecordingGraph& graph, Tensor& out, const Tensor& count)
        {
            OperandSet operands;
            operands.writes(out);
            graph.recorder().record_late_pair(TaskDescription(), operands, "test", count,
                                              static_cast<std::size_t>(out.num_elements),
                                              [] {}, [](std::size_t) {});
        }
    } // namespace Test
} // namespace SushiBLAS
