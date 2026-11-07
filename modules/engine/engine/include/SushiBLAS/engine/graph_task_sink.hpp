/**************************************************************************/
/* graph_task_sink.hpp                                                    */
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
 * @file graph_task_sink.hpp
 * @brief Declares GraphTaskSink, the TaskSink and LateCountSink that adds each task to a
 *        SushiRuntime::Recorder.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <cstddef>
#include <SushiBLAS/cost/cost.hpp>
#include <SushiBLAS/engine/task_sink.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/late_count_sink.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    /**
     * @brief Adds every task, and every late-count kernel, as a node of one SushiRuntime::Recorder.
     * @see modules/engine/engine/README.md
     */
    class GraphTaskSink final : public TaskSink, public LateCountSink
    {
        public:
            /**
             * @brief Binds the sink to the recorder it adds nodes to.
             * @param recorder Must outlive the sink.
             */
            explicit GraphTaskSink(SushiRuntime::Recorder& recorder) noexcept : recorder_(&recorder)
            {

            }

            /** @brief Reports true: the runtime runs a task of every mode. */
            [[nodiscard]] bool admits(ExecutionMode) const noexcept override { return true; }

            /** @brief Returns how many nodes the recorder holds. */
            [[nodiscard]] std::size_t node_count() const noexcept { return recorder_->node_count(); }

            /**
             * @brief Adds a node that runs the body on the runtime's queue, costed by the table
             *        and the operands' bytes, and keeps the operand storages until the node goes.
             * @throws std::runtime_error with the runtime's message when the recorder refuses.
             */
            void add_pair(const TaskDescription& description, const OperandSet& operands,
                          ExecutionMode mode, const std::string& origin, PairBody body) override;

            /**
             * @brief Adds the node record adds, under the description and the cost.
             * @throws std::runtime_error with the runtime's message when the recorder refuses.
             */
            void add_late_node(const TaskDescription& description, const Cost& cost,
                               const std::string& origin, const NodeRecord& record) override;

        private:
            /// @brief The graph recorder every task is added to; never null.
            SushiRuntime::Recorder* recorder_;
    };
} // namespace SushiBLAS
