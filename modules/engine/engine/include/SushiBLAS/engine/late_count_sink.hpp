/**************************************************************************/
/* late_count_sink.hpp                                                    */
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


/**
 * @file late_count_sink.hpp
 * @brief Declares LateCountSink, the port a TaskRecorder inserts a late-count graph node through.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <functional>
#include <SushiBLAS/cost/cost.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    struct TaskDescription;

    /**
     * @brief Receives a late-count kernel, a graph node whose count a previous node writes.
     *
     * A late-count kernel has no host form, so only a sink over a SushiRuntime graph takes one.
     * A TaskRecorder holds this port beside its TaskSink, or none; GraphTaskSink is its adapter.
     * @see modules/engine/engine/README.md
     */
    class LateCountSink
    {
        public:
            /// @brief Adds exactly one node to the recorder under the options it is given.
            using NodeRecord = std::function<void(SushiRuntime::Recorder&,
                                                  const SushiRuntime::NodeOptions&)>;

            LateCountSink(const LateCountSink&) = delete;
            LateCountSink& operator=(const LateCountSink&) = delete;
            LateCountSink(LateCountSink&&) = delete;
            LateCountSink& operator=(LateCountSink&&) = delete;
            virtual ~LateCountSink() = default;

            /**
             * @brief Adds the node record adds, under the name, operation, determinism,
             *        parameters and cost of the description.
             * @throws std::runtime_error with the runtime's message when the recorder refuses.
             */
            virtual void add_late_node(const TaskDescription& description, const Cost& cost,
                                       const std::string& origin, const NodeRecord& record) = 0;

        protected:
            /** @brief Lets an adapter construct itself; the port is never built alone. */
            LateCountSink() = default;
    };
} // namespace SushiBLAS
