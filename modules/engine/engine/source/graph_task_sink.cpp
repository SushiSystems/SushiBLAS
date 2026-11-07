/**************************************************************************/
/* graph_task_sink.cpp                                                    */
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
 * @file graph_task_sink.cpp
 * @brief Defines GraphTaskSink: the node that runs a recorded body and the insertion every node
 *        goes through.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/graph_task_sink.hpp>

#include <vector>
#include <cstdint>
#include <utility>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/pair_run.hpp>
#include <SushiBLAS/engine/task_cost.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>

namespace SushiBLAS
{
    namespace
    {
        /** @brief Runs a recorded body on the runtime's queue with a dispatcher that holds it. */
        class GraphPairTask
        {
            public:
                /** @brief Holds the body, the mode its dispatcher routes by and the operand storages. */
                GraphPairTask(TaskSink::PairBody body, ExecutionMode mode,
                              Internal::StorageKeepAlive retained)
                    : body_(std::move(body)), mode_(mode), retained_(std::move(retained))
                {

                }

                /** @brief Runs the body on @p q and returns the event that ends its work. */
                sycl::event operator()(sycl::queue& q, const std::vector<sycl::event>& deps) const
                {
                    const KernelPairDispatcher dispatcher(mode_, q);
                    PairRun run(dispatcher, deps);
                    body_(run);
                    return run.finished();
                }

            private:
                /// @brief The body the operation recorded.
                TaskSink::PairBody body_;

                /// @brief The mode the dispatcher routes by.
                ExecutionMode mode_;

                /// @brief The storages that stay allocated until the task is destroyed.
                Internal::StorageKeepAlive retained_;
        };

        /**
         * @brief Adds the node record adds to recorder under the options of the description.
         * @throws std::runtime_error with the runtime's message when the recorder refuses.
         */
        void insert_node(SushiRuntime::Recorder& recorder, const TaskDescription& description,
                         const Cost& cost, const std::string& origin,
                         const LateCountSink::NodeRecord& record)
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
            const std::size_t recorded_before = recorder.node_count();
            record(recorder, options);
            if (recorder.node_count() == recorded_before)
            {
                const SushiRuntime::Error* refusal = recorder.refusal();
                SB_THROW_IF(true, "{}: the runtime refused the recording: {}",
                            description.name,
                            refusal != nullptr ? refusal->message.c_str() : "no reason kept");
            }
        }
    } // namespace

    void GraphTaskSink::add_pair(const TaskDescription& description, const OperandSet& operands,
                                 ExecutionMode mode, const std::string& origin, PairBody body)
    {
        insert_node(*recorder_, description,
                    Internal::task_cost(description.operation, description.shape, operands),
                    origin,
                    [&](SushiRuntime::Recorder& recorder, const SushiRuntime::NodeOptions& options)
                    {
                        recorder.task(Internal::task_access(operands),
                                      GraphPairTask(std::move(body), mode, operands.retained()),
                                      options);
                    });
    }

    void GraphTaskSink::add_late_node(const TaskDescription& description, const Cost& cost,
                                      const std::string& origin, const NodeRecord& record)
    {
        insert_node(*recorder_, description, cost, origin, record);
    }
} // namespace SushiBLAS
