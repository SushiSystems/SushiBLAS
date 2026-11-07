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
#include <functional>
#include <SushiBLAS/engine/execution_mode.hpp>

namespace SushiBLAS
{
    class OperandSet;
    class PairRun;
    struct TaskDescription;

    /**
     * @brief Receives the kernel-pair tasks a TaskRecorder records, one call at a time under
     *        the recorder's mutex.
     *
     * It names no SYCL type, so a host consumer includes it without SYCL; PairRun carries the
     * dispatcher and the events. GraphTaskSink and HostTaskSink are its adapters.
     * @see modules/engine/engine/README.md
     */
    class TaskSink
    {
        public:
            /// @brief A recorded body: reads its dispatcher and dependencies from the run and finishes it with the event that ends its work.
            using PairBody = std::function<void(PairRun&)>;

            TaskSink(const TaskSink&) = delete;
            TaskSink& operator=(const TaskSink&) = delete;
            TaskSink(TaskSink&&) = delete;
            TaskSink& operator=(TaskSink&&) = delete;
            virtual ~TaskSink() = default;

            /** @brief Reports whether tasks recorded under @p mode can run through this sink. */
            [[nodiscard]] virtual bool admits(ExecutionMode mode) const noexcept = 0;

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
