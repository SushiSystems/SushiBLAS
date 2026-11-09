/**************************************************************************/
/* host_graph_host.hpp                                                    */
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
 * @file host_graph_host.hpp
 * @brief Declares HostGraphHost, the GraphHost that keeps its tasks on the host and runs them in ForceReference.
 * @author Mustafa Garip
 * @see modules/engine/recording_graph/README.md
 */

#pragma once

#include <string>
#include <utility>
#include <optional>
#include <cstddef>
#include <SushiBLAS/engine/host_task_sink.hpp>
#include <SushiBLAS/recording_graph/graph_host.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    /**
     * @brief Keeps the recorded tasks in a HostTaskSink, so a graph records and runs without a runtime.
     *
     * It admits ForceReference only, takes no late-count node and holds no late-count storage:
     * a late-count task is a host body there. It has no plan, and the label names nothing.
     */
    class HostGraphHost final : public GraphHost
    {
        public:
            [[nodiscard]] TaskSink& sink() noexcept override { return sink_; }

            [[nodiscard]] LateCountSink* late_sink() noexcept override { return nullptr; }

            [[nodiscard]] LateCountHold* late_counts() noexcept override { return nullptr; }

            [[nodiscard]] std::size_t node_count() const noexcept override { return sink_.recorded_count(); }

            [[nodiscard]] bool has_refusal() const noexcept override { return false; }

            [[nodiscard]] SushiRuntime::Result<SushiRuntime::PlanView> plan() override
            {
                return SushiRuntime::make_error(SushiRuntime::Error(
                    SushiRuntime::ErrorCode::Unsupported, "a host graph has no plan"));
            }

            [[nodiscard]] SushiRuntime::Result<void> run() override
            {
                const std::optional<HostTaskFailure> failure = sink_.replay();
                if (!failure)
                    return {};
                return SushiRuntime::make_error(SushiRuntime::Error(
                    SushiRuntime::ErrorCode::KernelFailed,
                    "task " + std::to_string(failure->index) + " '" + failure->name + "' failed: " +
                        failure->message));
            }

            void clear() override { static_cast<void>(sink_.discard()); }

            void set_label(std::string) override
            {

            }

        private:
            /// @brief The tasks, in record order.
            HostTaskSink sink_;
    };
} // namespace SushiBLAS
