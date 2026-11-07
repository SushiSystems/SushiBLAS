/**************************************************************************/
/* task_sink_subjects.hpp                                                 */
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
 * @file task_sink_subjects.hpp
 * @brief Declares the subjects a TaskSink contract runs on: a TaskRecorder over the host sink and
 *        one over the graph adapter on the shared runtime.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <optional>
#include <cstdint>
#include <vector>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/host_task_sink.hpp>
#include <SushiBLAS/engine/runtime_scratch_source.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/recording_graph/recording_graph.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiRuntime/core/span.hpp>
#include "support/tensor_subjects.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The number of elements of the tensor every contract task writes.
        inline constexpr std::int64_t K_SINK_TARGET_ELEMENTS = 4;

        /**
         * @brief Returns a FLOAT32 vector from a tensor subject's factory.
         *
         * Every task of a contract writes it, so the graph orders the tasks as they were recorded.
         */
        template<typename TensorSubject>
        [[nodiscard]] Tensor sink_target_of(const TensorSubject& tensors)
        {
            const std::vector<std::int64_t> dims{K_SINK_TARGET_ELEMENTS};
            return tensors.factory.create(
                SushiRuntime::span<const std::int64_t>(dims.data(), dims.size()),
                Core::DataType::Float32);
        }

        /** @brief Records on a host sink, runs the bodies on the host, and has no queue. */
        class HostSinkSubject
        {
            public:
                /// @brief Whether the dispatcher a body gets holds a queue.
                static constexpr bool K_HAS_QUEUE = false;

                /** @brief Builds a recorder in ForceReference over an empty host sink. */
                HostSinkSubject() : recorder_(sink_), target_(sink_target_of(tensors_))
                {

                }

                HostSinkSubject(const HostSinkSubject&) = delete;
                HostSinkSubject& operator=(const HostSinkSubject&) = delete;

                /** @brief Returns the recorder under test. */
                [[nodiscard]] TaskRecorder& recorder() noexcept { return recorder_; }

                /** @brief Returns the tensor every task writes. */
                [[nodiscard]] const Tensor& target() const noexcept { return target_; }

                /**
                 * @brief Runs the recorded tasks.
                 * @return Empty when all ran, else the name of the failed task and its message.
                 */
                [[nodiscard]] std::string run()
                {
                    const std::optional<HostTaskFailure> failure = sink_.run();
                    if (!failure.has_value())
                        return std::string();
                    return failure->name + ": " + failure->message;
                }

            private:
                /// @brief Makes the target tensor.
                HostTensorSubject tensors_;

                /// @brief Keeps the bodies.
                HostTaskSink sink_;

                /// @brief The recorder under test.
                TaskRecorder recorder_;

                /// @brief The tensor every task writes.
                Tensor target_;
        };

        /** @brief Records on a runtime graph through the graph adapter, in ForceReference. */
        class GraphSinkSubject
        {
            public:
                /// @brief Whether the dispatcher a body gets holds a queue.
                static constexpr bool K_HAS_QUEUE = true;

                /** @brief Builds a graph on the shared runtime and sets the recorder to ForceReference. */
                GraphSinkSubject() : target_(sink_target_of(tensors_))
                {
                    graph_.recorder().set_mode(ExecutionMode::ForceReference);
                }

                GraphSinkSubject(const GraphSinkSubject&) = delete;
                GraphSinkSubject& operator=(const GraphSinkSubject&) = delete;

                /** @brief Returns the recorder under test. */
                [[nodiscard]] TaskRecorder& recorder() noexcept { return graph_.recorder(); }

                /** @brief Returns the tensor every task writes. */
                [[nodiscard]] const Tensor& target() const noexcept { return target_; }

                /**
                 * @brief Runs the recorded tasks.
                 * @return Empty when all ran, else the runtime's message for the failed run.
                 */
                [[nodiscard]] std::string run()
                {
                    const SushiRuntime::Result<void> ran = graph_.run();
                    if (ran)
                        return std::string();
                    return ran.error().message;
                }

            private:
                /// @brief Makes the target tensor.
                RuntimeTensorSubject tensors_;

                /// @brief Lends device memory to the pool from the test runtime.
                RuntimeScratchSource source_{TestEnvironment::runtime()};

                /// @brief Lends scratch to the recorder.
                ScratchPool pool_{source_};

                /// @brief The graph and the recorder over its adapter.
                RecordingGraph graph_{TestEnvironment::runtime(), pool_};

                /// @brief The tensor every task writes.
                Tensor target_;
        };
    } // namespace Test
} // namespace SushiBLAS
