/**************************************************************************/
/* task_sink.hpp                                                          */
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
 * @file task_sink.hpp
 * @brief Declares TaskSink, the port TaskRecorder records its tasks through.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <vector>
#include <functional>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>

namespace SushiBLAS
{
    class GraphTaskSink;

    /**
     * @brief Receives the kernel-pair tasks a TaskRecorder records.
     *
     * GraphTaskSink adds each task as a node of a SushiRuntime graph. HostTaskSink keeps the
     * bodies in record order and runs them on the host under ForceReference. TaskRecorder calls a
     * sink with its recording mutex held, one call at a time.
     * @see modules/engine/engine/README.md
     */
    class TaskSink
    {
        public:
            /// @brief A recorded body: gets its dispatcher and dependencies, returns the event that ends its work.
            using PairBody = std::function<sycl::event(const KernelPairDispatcher&,
                                                       const std::vector<sycl::event>&)>;

            TaskSink(const TaskSink&) = delete;
            TaskSink& operator=(const TaskSink&) = delete;
            TaskSink(TaskSink&&) = delete;
            TaskSink& operator=(TaskSink&&) = delete;
            virtual ~TaskSink() = default;

            /** @brief Reports whether tasks recorded under @p mode can run through this sink. */
            [[nodiscard]] virtual bool admits(ExecutionMode mode) const noexcept = 0;

            /**
             * @brief Reports whether the dispatcher this sink hands a body holds a queue.
             *
             * A body of the old shape `(dispatcher, sycl::queue&, deps)` needs one; TaskRecorder
             * refuses it at record time on a sink that provides none.
             */
            [[nodiscard]] virtual bool provides_queue() const noexcept = 0;

            /**
             * @brief Returns the graph adapter behind this sink, or null when none is.
             *
             * A late-count kernel is a graph node with no host form, so TaskRecorder reaches it
             * only through this.
             */
            [[nodiscard]] virtual GraphTaskSink* graph() noexcept = 0;

            /**
             * @brief Adds one task whose body runs a kernel pair.
             * @param mode The mode the task was recorded under; its dispatcher routes by it.
             * @param origin The label the task carries as its origin.
             * @param body Runs the task; the sink hands it a dispatcher fixed to @p mode.
             * @throws std::runtime_error when the sink refuses the task; none is added.
             */
            virtual void add_pair(const TaskDescription& description, const OperandSet& operands,
                                  ExecutionMode mode, const std::string& origin,
                                  PairBody body) = 0;

        protected:
            /** @brief Lets an adapter construct itself; the port is never built alone. */
            TaskSink() = default;
    };
} // namespace SushiBLAS
