/**************************************************************************/
/* runtime_graph_host.hpp                                                 */
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
 * @file runtime_graph_host.hpp
 * @brief Declares RuntimeGraphHost, the GraphHost that records on a SushiRuntime graph.
 * @author Mustafa Garip
 * @see modules/engine/recording_graph/README.md
 */

#pragma once

#include <string>
#include <utility>
#include <cstddef>
#include <SushiBLAS/engine/graph_task_sink.hpp>
#include <SushiBLAS/engine/late_count_hold.hpp>
#include <SushiBLAS/recording_graph/graph_host.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    /** @brief Owns a runtime graph, the sink over its recorder and the hold of its late-count storages. */
    class RuntimeGraphHost final : public GraphHost
    {
        public:
            /**
             * @brief Builds an empty graph on a runtime.
             * @param runtime Must outlive this object.
             */
            explicit RuntimeGraphHost(SushiRuntime::Runtime& runtime)
                : graph_(runtime.graph()), sink_(graph_.recorder()), late_counts_(runtime)
            {

            }

            [[nodiscard]] TaskSink& sink() noexcept override { return sink_; }

            [[nodiscard]] LateCountSink* late_sink() noexcept override { return &sink_; }

            [[nodiscard]] LateCountHold* late_counts() noexcept override { return &late_counts_; }

            [[nodiscard]] std::size_t node_count() const noexcept override
            {
                return graph_.recorder().node_count();
            }

            [[nodiscard]] bool has_refusal() const noexcept override
            {
                return graph_.recorder().refusal() != nullptr;
            }

            [[nodiscard]] SushiRuntime::Result<SushiRuntime::PlanView> plan() override
            {
                return graph_.plan();
            }

            [[nodiscard]] SushiRuntime::Result<void> run() override
            {
                SushiRuntime::Result<SushiRuntime::RunReport*> ran = graph_.run(report_);
                if (ran)
                    return {};
                return SushiRuntime::make_error(std::move(ran.error()));
            }

            void clear() override
            {
                graph_.clear();
                late_counts_.release();
            }

            void set_label(std::string label) override
            {
                graph_.set_label(std::move(label));
            }

        private:
            /// @brief The graph recorded tasks wait in.
            SushiRuntime::Graph graph_;

            /// @brief The sink that adds tasks to the recorder of graph_.
            GraphTaskSink sink_;

            /// @brief The storages of graph_'s late-count tasks, released when graph_ is emptied.
            LateCountHold late_counts_;

            /// @brief The report every run writes into, so a run allocates none.
            SushiRuntime::RunReport report_;
    };
} // namespace SushiBLAS
