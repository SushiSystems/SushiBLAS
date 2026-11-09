/**************************************************************************/
/* graph_host.hpp                                                         */
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
 * @file graph_host.hpp
 * @brief Declares GraphHost, the port a RecordingGraph records into and runs through.
 * @author Mustafa Garip
 * @see modules/engine/recording_graph/README.md
 */

#pragma once

#include <string>
#include <cstddef>
#include <SushiBLAS/engine/late_count_hold.hpp>
#include <SushiBLAS/engine/late_count_sink.hpp>
#include <SushiBLAS/engine/task_sink.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    /**
     * @brief Holds the tasks a RecordingGraph records and runs them.
     *
     * RuntimeGraphHost runs them on a runtime graph and HostGraphHost on the host, so a test of
     * the graph's own logic needs no device runtime. A host that keeps its graph elsewhere adds
     * one more adapter.
     * @see modules/engine/recording_graph/README.md
     */
    class GraphHost
    {
        public:
            GraphHost(const GraphHost&) = delete;
            GraphHost& operator=(const GraphHost&) = delete;
            GraphHost(GraphHost&&) = delete;
            GraphHost& operator=(GraphHost&&) = delete;
            virtual ~GraphHost() = default;

            /** @brief Returns the sink a TaskRecorder adds its tasks to. */
            [[nodiscard]] virtual TaskSink& sink() noexcept = 0;

            /** @brief Returns the sink that takes late-count nodes, or null when the host takes none. */
            [[nodiscard]] virtual LateCountSink* late_sink() noexcept = 0;

            /** @brief Returns the hold that keeps late-count storages, or null when the host needs none. */
            [[nodiscard]] virtual LateCountHold* late_counts() noexcept = 0;

            /** @brief Returns how many tasks are recorded and have not run. */
            [[nodiscard]] virtual std::size_t node_count() const noexcept = 0;

            /** @brief Reports whether the recording was refused and a run would report it. */
            [[nodiscard]] virtual bool has_refusal() const noexcept = 0;

            /** @brief Returns what the plan orders, or the error that kept it from existing. */
            [[nodiscard]] virtual SushiRuntime::Result<SushiRuntime::PlanView> plan() = 0;

            /** @brief Runs the recorded tasks and keeps them; returns the error that ended the run. */
            [[nodiscard]] virtual SushiRuntime::Result<void> run() = 0;

            /** @brief Drops every recorded task, with what its body captured, and the storages late-count tasks held. */
            virtual void clear() = 0;

            /** @brief Names the graph in the spans of its runs. */
            virtual void set_label(std::string label) = 0;

        protected:
            /** @brief Lets an adapter construct itself; the port is never built alone. */
            GraphHost() = default;
    };
} // namespace SushiBLAS
