/**************************************************************************/
/* recorded_graph.hpp                                                     */
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
 * @file recorded_graph.hpp
 * @brief Declares RecordedGraph, the one owner of the graph recorded tasks wait in.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <utility>
#include <SushiRuntime/api/graph/graph.hpp>
#include <SushiRuntime/api/graph/plan_view.hpp>
#include <SushiRuntime/api/graph/recorder.hpp>
#include <SushiRuntime/api/runtime/runtime.hpp>
#include <SushiRuntime/core/error.hpp>
#include <SushiRuntime/core/result.hpp>
#include <SushiRuntime/report/run_report.hpp>

namespace SushiBLAS
{
    /**
     * @brief Owns one SushiRuntime::Graph, runs what is recorded on it and empties it.
     * @see modules/engine/engine/README.md
     */
    class RecordedGraph final
    {
        public:
            /**
             * @brief Builds an empty graph on a runtime.
             * @param runtime Must outlive this object.
             */
            explicit RecordedGraph(SushiRuntime::Runtime& runtime)
                : runtime_(&runtime), graph_(runtime.graph())
            {

            }

            RecordedGraph(const RecordedGraph&) = delete;
            RecordedGraph& operator=(const RecordedGraph&) = delete;
            RecordedGraph(RecordedGraph&&) = delete;
            RecordedGraph& operator=(RecordedGraph&&) = delete;
            ~RecordedGraph() = default;

            /**
             * @brief Returns the recorder tasks are recorded on.
             * @return A reference that stays valid across run() and discard().
             */
            [[nodiscard]] SushiRuntime::Recorder& recorder() noexcept
            {
                return graph_.recorder();
            }

            /** @brief Returns the recorder tasks are recorded on, read-only. */
            [[nodiscard]] const SushiRuntime::Recorder& recorder() const noexcept
            {
                return graph_.recorder();
            }

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
             * @brief Runs the recorded tasks, returns when every one has finished, and leaves
             *        the graph empty.
             * @return Nothing, also when nothing is recorded, or the error that ended the run.
             *         After InvalidState for a run in flight the recorded tasks are kept.
             * @pre No thread is recording on recorder().
             * @see modules/engine/engine/README.md
             */
            [[nodiscard]] SushiRuntime::Result<void> run()
            {
                if (is_empty())
                    return {};

                const bool recording_refused = graph_.recorder().refusal() != nullptr;
                SushiRuntime::Result<SushiRuntime::RunReport*> ran = graph_.run(report_);
                if (ran)
                {
                    leave_empty();
                    return {};
                }

                // A run the runtime turned away for one in flight started no recorded task.
                const bool turned_away = !recording_refused &&
                                         ran.error().code == SushiRuntime::ErrorCode::InvalidState;
                if (!turned_away)
                    leave_empty();

                return SushiRuntime::make_error(std::move(ran.error()));
            }

            /**
             * @brief Drops every recorded task with what its body captured, and the plan.
             * @return How many tasks were dropped.
             * @pre No thread is recording on recorder() or running run().
             */
            std::size_t discard()
            {
                const std::size_t dropped = recorded_count();
                leave_empty();
                return dropped;
            }

        private:
            /** @brief Replaces the graph with an empty one, so its tasks and its plan die. */
            void leave_empty()
            {
                graph_ = runtime_->graph();
            }

            /// @brief The runtime every graph of this object runs on; never null.
            SushiRuntime::Runtime* runtime_;

            /// @brief The graph recorded tasks wait in.
            SushiRuntime::Graph graph_;

            /// @brief The report every run writes into, so a run allocates none.
            SushiRuntime::RunReport report_;
    };
} // namespace SushiBLAS
