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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file recording_graph.hpp
 * @brief Declares RecordingGraph, a runtime graph with a TaskRecorder recording on it.
 * @author Mustafa Garip
 * @see modules/engine/recording_graph/README.md
 */

#pragma once

#include <cstddef>
#include <utility>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/late_count_hold.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    /// @brief The execution mode a RecordingGraph records under unless its caller sets another.
    constexpr ExecutionMode K_DEFAULT_RECORDING_MODE = ExecutionMode::Auto;

    /**
     * @brief Owns a runtime graph and a TaskRecorder over its recorder, the way a caller that
     *        records a few operations does.
     *
     * It includes no test framework, so a test fixture and a benchmark program share it.
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
                : graph_(runtime.graph()), late_counts_(runtime),
                  recorder_(graph_.recorder(), scratch_pool, late_counts_, mode)
            {

            }

            /**
             * @brief Builds an empty graph on an engine's runtime and a recorder over it that
             *        borrows the engine's scratch pool and execution mode.
             * @param engine Must outlive this object.
             */
            explicit RecordingGraph(Engine& engine)
                : RecordingGraph(engine.runtime(), engine.scratch_pool(), engine.mode())
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

            /** @brief Returns the hold that keeps the storages of this graph's late-count tasks. */
            [[nodiscard]] const LateCountHold& late_counts() const noexcept { return late_counts_; }

            /** @brief Returns how many tasks are recorded and have not run. */
            [[nodiscard]] std::size_t recorded_count() const noexcept
            {
                return graph_.recorder().node_count();
            }

            /** @brief Reports whether a run would have nothing to do and nothing to report. */
            [[nodiscard]] bool is_empty() const noexcept
            {
                return recorded_count() == 0 && graph_.recorder().refusal() == nullptr;
            }

            /**
             * @brief Returns what the plan orders, building the plan when none is current.
             * @return One entry per recorded task in record order, or the error that kept the
             *         graph from compiling.
             */
            [[nodiscard]] SushiRuntime::Result<SushiRuntime::PlanView> plan()
            {
                return graph_.plan();
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

                const bool recording_refused = graph_.recorder().refusal() != nullptr;
                SushiRuntime::Result<SushiRuntime::RunReport*> ran = graph_.run(report_);
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
             * @brief Empties the graph while no record call adds a task, and lets go of the
             *        storages its late-count tasks held.
             */
            void forget()
            {
                recorder_.run_exclusive(
                    [this]()
                    {
                        graph_.clear();
                        late_counts_.release();
                    });
            }

            /// @brief The graph recorded tasks wait in.
            SushiRuntime::Graph graph_;

            /// @brief The storages of graph_'s late-count tasks, released when graph_ is emptied.
            LateCountHold late_counts_;

            /// @brief The recorder bound to the recorder of graph_, a scratch pool and late_counts_.
            TaskRecorder recorder_;

            /// @brief The report every run writes into, so a run allocates none.
            SushiRuntime::RunReport report_;
    };
} // namespace SushiBLAS
