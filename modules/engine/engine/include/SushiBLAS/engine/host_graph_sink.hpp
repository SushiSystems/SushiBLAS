/**************************************************************************/
/* host_graph_sink.hpp                                                    */
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
 * @file host_graph_sink.hpp
 * @brief Declares HostGraphSink, the adapter of both sink ports that admits every mode and keeps what each task stated, with no runtime.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <utility>
#include <SushiBLAS/cost/cost.hpp>
#include <SushiBLAS/engine/host_task_sink.hpp>
#include <SushiBLAS/engine/late_count_sink.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/engine/task_sink.hpp>

namespace SushiBLAS
{
    /** @brief Holds what a late-count node stated when it was added: its name, cost and origin label. */
    struct HostLateNode
    {
        /// @brief The name of the description of the node.
        std::string name;

        /// @brief The cost the node stated.
        Cost cost;

        /// @brief The origin label the node carried.
        std::string origin;
    };

    /**
     * @brief Records a late-count task under any mode, as GraphTaskSink does, and keeps what it stated.
     *
     * It admits every mode, so a recorder over it takes the path that adds a late-count node. A
     * pair is kept by a HostTaskSink, which runs it under ForceReference only; a late node is kept
     * as a HostLateNode and its record is not called, since the node has no host form.
     * @see modules/engine/engine/README.md
     */
    class HostGraphSink final : public TaskSink, public LateCountSink
    {
        public:
            /** @brief Holds no task. */
            HostGraphSink() = default;

            /** @brief Reports true for every mode. */
            [[nodiscard]] bool admits(ExecutionMode) const noexcept override { return true; }

            /** @brief Keeps the pair as the HostTaskSink it holds does; the mode is not kept. */
            void add_pair(const TaskDescription& description, const OperandSet& operands,
                          ExecutionMode, const std::string& origin, PairBody body) override
            {
                pairs_.add_pair(description, operands, ExecutionMode::ForceReference, origin,
                                std::move(body));
            }

            /** @brief Keeps the name, cost and origin of the node; record is not called. */
            void add_late_node(const TaskDescription& description, const Cost& cost,
                               const std::string& origin, const NodeRecord&) override
            {
                late_nodes_.push_back(HostLateNode{description.name, cost, origin});
            }

            /** @brief Returns the sink that keeps the pairs. */
            [[nodiscard]] const HostTaskSink& pairs() const noexcept { return pairs_; }

            /** @brief Returns what each late-count node stated, in the order they were added. */
            [[nodiscard]] const std::vector<HostLateNode>& late_nodes() const noexcept
            {
                return late_nodes_;
            }

        private:
            /// @brief Keeps the pairs.
            HostTaskSink pairs_;

            /// @brief The late-count nodes, in the order they were added.
            std::vector<HostLateNode> late_nodes_;
    };
} // namespace SushiBLAS
