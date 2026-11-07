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
 * @brief Declares GraphTaskSink, the TaskSink that adds each task to a SushiRuntime::Recorder.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/cost/cost.hpp>
#include <SushiBLAS/engine/task_cost.hpp>
#include <SushiBLAS/engine/task_sink.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        namespace Detail
        {
            /** @brief Runs a recorded body on the runtime's queue with a dispatcher that holds it. */
            class GraphPairTask
            {
                public:
                    /** @brief Holds the body, the mode its dispatcher routes by and the operand storages. */
                    GraphPairTask(TaskSink::PairBody body, ExecutionMode mode,
                                  StorageKeepAlive retained)
                        : body_(std::move(body)), mode_(mode), retained_(std::move(retained))
                    {

                    }

                    /** @brief Runs the body on @p q and returns the event that ends its work. */
                    sycl::event operator()(sycl::queue& q, const std::vector<sycl::event>& deps) const
                    {
                        return body_(KernelPairDispatcher(mode_, q), deps);
                    }

                private:
                    /// @brief The body the operation recorded.
                    TaskSink::PairBody body_;

                    /// @brief The mode the dispatcher routes by.
                    ExecutionMode mode_;

                    /// @brief The storages that stay allocated until the task is destroyed.
                    StorageKeepAlive retained_;
            };
        } // namespace Detail
    } // namespace Internal

    /**
     * @brief Adds every task as a node of one SushiRuntime::Recorder.
     * @see modules/engine/engine/README.md
     */
    class GraphTaskSink final : public TaskSink
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

            /** @brief Returns this sink. */
            [[nodiscard]] GraphTaskSink* graph() noexcept override { return this; }

            /** @brief Returns how many nodes the recorder holds. */
            [[nodiscard]] std::size_t node_count() const noexcept { return recorder_->node_count(); }

            /**
             * @brief Adds a node that runs the body, costed by the table and the operands' bytes.
             * @throws std::runtime_error with the runtime's message when the recorder refuses.
             */
            void add_pair(const TaskDescription& description, const OperandSet& operands,
                          ExecutionMode mode, const std::string& origin, PairBody body) override
            {
                insert_node(
                    description,
                    Internal::task_cost(description.operation, description.shape, operands),
                    origin,
                    [&](SushiRuntime::Recorder& recorder, const SushiRuntime::NodeOptions& options)
                    {
                        recorder.task(Internal::task_access(operands),
                                      Internal::Detail::GraphPairTask(std::move(body), mode,
                                                                      operands.retained()),
                                      options);
                    });
            }

            /**
             * @brief Adds one node under the name, the operation, the determinism, the
             *        parameters and the cost.
             * @param cost What the node states about its work; an uncosted cost states none.
             * @param record Callable `(SushiRuntime::Recorder&, const SushiRuntime::NodeOptions&)`
             *        that adds exactly one node.
             * @throws std::runtime_error with the runtime's message when the recorder refuses.
             */
            template<typename Record>
            void insert_node(const TaskDescription& description, const Cost& cost,
                             const std::string& origin, Record&& record)
            {
                static_assert(TaskDescription::K_MAX_PARAMETERS <=
                                  SushiRuntime::NodeOptions::K_MAX_PARAMETERS,
                              "A task description carries more parameters than a runtime node "
                              "takes.");
                SushiRuntime::NodeOptions options;
                options.name = description.name;
                options.operation = description.operation;
                options.determinism = description.determinism;
                options.parameters = SushiRuntime::span<const std::uint64_t>(
                    description.parameters, TaskDescription::K_MAX_PARAMETERS);
                options.cost = Internal::node_cost_of(cost);
                options.origin = origin;

                // The count before and the recording are one step under the recorder's mutex.
                const std::size_t recorded_before = recorder_->node_count();
                std::forward<Record>(record)(*recorder_, options);
                if (recorder_->node_count() == recorded_before)
                {
                    const SushiRuntime::Error* refusal = recorder_->refusal();
                    SB_THROW_IF(true, "{}: the runtime refused the recording: {}",
                                description.name,
                                refusal != nullptr ? refusal->message.c_str()
                                                   : "no reason kept");
                }
            }

        private:
            /// @brief The graph recorder every task is added to; never null.
            SushiRuntime::Recorder* recorder_;
    };
} // namespace SushiBLAS
