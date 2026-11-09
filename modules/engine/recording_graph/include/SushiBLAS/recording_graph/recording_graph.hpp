/**************************************************************************/
/* recording_graph.hpp                                                    */
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
 * @file recording_graph.hpp
 * @brief Declares RecordingGraph, a runtime graph with a TaskRecorder recording on it.
 * @author Mustafa Garip
 * @see modules/engine/recording_graph/README.md
 */

#pragma once

#include <memory>
#include <cstddef>
#include <string>
#include <utility>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/late_count_hold.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/recording_graph/graph_host.hpp>
#include <SushiBLAS/recording_graph/runtime_graph_host.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    /// @brief The execution mode a RecordingGraph records under unless its caller sets another.
    constexpr ExecutionMode K_DEFAULT_RECORDING_MODE = ExecutionMode::Auto;

    /**
     * @brief Owns a GraphHost and a TaskRecorder over its sink, the way a caller that records
     *        a few operations does.
     *
     * A runtime builds a runtime graph; a HostGraphHost records and runs on the host. It
     * includes no test framework, so a test fixture and a benchmark program share it.
     * @see modules/engine/recording_graph/README.md
     */
    class RecordingGraph final
    {
        public:
            /**
             * @brief Builds an empty graph on a runtime and a recorder over it.
             * @param runtime Must outlive this object.
             * @param scratch_pool Lends operation scratch; it must outlive this object.
             * @param mode The execution mode the recorder records under.
             */
            RecordingGraph(SushiRuntime::Runtime& runtime, ScratchPool& scratch_pool,
                           ExecutionMode mode = K_DEFAULT_RECORDING_MODE)
                : RecordingGraph(std::make_unique<RuntimeGraphHost>(runtime), scratch_pool, mode)
            {

            }

            /**
             * @brief Builds an empty graph over a host.
             * @param host Holds the tasks and runs them; this object owns it.
             * @param scratch_pool Lends operation scratch; it must outlive this object.
             * @param mode The execution mode the recorder records under.
             * @throws UnsupportedExecution when the host does not admit the mode.
             */
            RecordingGraph(std::unique_ptr<GraphHost> host, ScratchPool& scratch_pool,
                           ExecutionMode mode = K_DEFAULT_RECORDING_MODE)
                : host_(std::move(host)),
                  recorder_(host_->sink(), &scratch_pool, host_->late_counts(), mode, host_->late_sink())
            {

            }

            RecordingGraph(const RecordingGraph&) = delete;
            RecordingGraph& operator=(const RecordingGraph&) = delete;
            RecordingGraph(RecordingGraph&&) = delete;
            RecordingGraph& operator=(RecordingGraph&&) = delete;
            ~RecordingGraph() = default;

            /** @brief Returns the recorder a view or a seam records on this graph through. */
            [[nodiscard]] TaskRecorder& recorder() noexcept { return recorder_; }

            /** @brief Returns the recorder a view or a seam records on this graph through. */
            [[nodiscard]] const TaskRecorder& recorder() const noexcept { return recorder_; }

            /**
             * @brief Returns the hold that keeps the storages of this graph's late-count tasks.
             * @return Null when the host holds none.
             */
            [[nodiscard]] const LateCountHold* late_counts() const noexcept { return host_->late_counts(); }

            /**
             * @brief Names the graph in the spans of its runs, so a profile can tell graphs apart.
             * @param label A short name such as the scenario a benchmark measures.
             */
            void set_label(std::string label)
            {
                host_->set_label(std::move(label));
            }

            /** @brief Returns how many tasks are recorded and have not run. */
            [[nodiscard]] std::size_t recorded_count() const noexcept
            {
                return host_->node_count();
            }

            /** @brief Reports whether a run would have nothing to do and nothing to report. */
            [[nodiscard]] bool is_empty() const noexcept
            {
                return recorded_count() == 0 && !host_->has_refusal();
            }

            /**
             * @brief Returns what the plan orders, building the plan when none is current.
             * @return One entry per recorded task in record order, or the error that kept the
             *         graph from compiling.
             */
            [[nodiscard]] SushiRuntime::Result<SushiRuntime::PlanView> plan()
            {
                return host_->plan();
            }

            /**
             * @brief Runs the recorded graph, returns when it has finished and leaves it empty,
             *        with the storages of its late-count tasks let go.
             * @return Nothing, also when no task was recorded, or the error that ended the run.
             *         After InvalidState for a run in flight the recorded tasks are kept.
             * @pre No thread is recording on recorder().
             */
            [[nodiscard]] SushiRuntime::Result<void> run()
            {
                if (is_empty())
                    return {};

                const bool recording_refused = host_->has_refusal();
                SushiRuntime::Result<void> ran = host_->run();
                if (ran)
                {
                    forget();
                    return {};
                }

                // A run the runtime turned away for one in flight started no recorded task.
                const bool turned_away =
                    !recording_refused &&
                    ran.error().code == SushiRuntime::ErrorCode::InvalidState;
                if (!turned_away)
                    forget();
                return SushiRuntime::make_error(std::move(ran.error()));
            }

            /**
             * @brief Drops every recorded task with what its body captured, the storages its
             *        late-count tasks held, and the plan.
             * @return How many tasks were dropped.
             * @pre No thread is running run().
             */
            std::size_t discard()
            {
                const std::size_t dropped = recorded_count();
                forget();
                return dropped;
            }

        private:
            /**
             * @brief Empties the host while no record call adds a task: it drops the tasks, frees
             *        the compiled plan and lets go of the storages the late-count tasks held.
             */
            void forget()
            {
                recorder_.run_exclusive(
                    [this]()
                    {
                        host_->clear();
                    });
            }

            /// @brief Holds the recorded tasks and runs them.
            std::unique_ptr<GraphHost> host_;

            /// @brief The recorder bound to the sink of host_, a scratch pool and the hold of host_.
            TaskRecorder recorder_;
    };
} // namespace SushiBLAS
