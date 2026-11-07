/**************************************************************************/
/* host_recording_fixture.hpp                                             */
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
 * @file host_recording_fixture.hpp
 * @brief Declares HostRecordingFixture, a TaskRecorder over a host engine's sink, with its GoogleTest helpers.
 * @author Mustafa Garip
 * @see modules/engine/engine_fixture/README.md
 */

#pragma once

#include <cstddef>
#include <optional>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/host_task_sink.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Holds a TaskRecorder over a host engine's HostTaskSink and adds the GoogleTest
         *        assertion on a run.
         *
         * It makes no queue and no graph. The recorded bodies run on the host, in record order,
         * under ForceReference.
         * @see modules/engine/engine_fixture/README.md
         */
        class HostRecordingFixture final
        {
            public:
                /**
                 * @brief Builds a recorder over the host engine's sink that borrows its scratch pool.
                 * @param engine A host engine; it must outlive the fixture.
                 * @throws UnsupportedExecution when the engine was built on a runtime.
                 */
                explicit HostRecordingFixture(Engine& engine)
                    : sink_(&engine.host_sink()),
                      recorder_(*sink_, &engine.scratch_pool(), nullptr, engine.mode())
                {

                }

                HostRecordingFixture(const HostRecordingFixture&) = delete;
                HostRecordingFixture& operator=(const HostRecordingFixture&) = delete;
                HostRecordingFixture(HostRecordingFixture&&) = delete;
                HostRecordingFixture& operator=(HostRecordingFixture&&) = delete;
                ~HostRecordingFixture() = default;

                /** @brief Returns the recorder a view or a seam records through. */
                [[nodiscard]] TaskRecorder& recorder() noexcept { return recorder_; }

                /** @brief Returns how many tasks are recorded and have not run. */
                [[nodiscard]] std::size_t recorded_count() const noexcept
                {
                    return sink_->recorded_count();
                }

                /** @brief Reports whether a run would have nothing to do. */
                [[nodiscard]] bool is_empty() const noexcept { return recorded_count() == 0; }

                /**
                 * @brief Runs the recorded bodies in record order and drops them, as HostTaskSink::run() does.
                 * @return Nothing when every body ran, else the first failure.
                 */
                [[nodiscard]] std::optional<HostTaskFailure> run() { return sink_->run(); }

                /**
                 * @brief Runs the recorded bodies as run() does and reports whether all succeeded.
                 * @return Success, or a failure that names the task and its message.
                 */
                [[nodiscard]] ::testing::AssertionResult run_succeeds()
                {
                    const std::optional<HostTaskFailure> failure = run();
                    if (!failure.has_value())
                        return ::testing::AssertionSuccess();
                    return ::testing::AssertionFailure()
                           << "task " << failure->index << " (" << failure->name
                           << ") failed: " << failure->message;
                }

                /** @brief Drops every recorded task without running it and returns how many were dropped. */
                std::size_t discard() noexcept { return sink_->discard(); }

            private:
                /// @brief The engine's sink; never null.
                HostTaskSink* sink_;

                /// @brief The recorder over sink_.
                TaskRecorder recorder_;
        };
    } // namespace Test
} // namespace SushiBLAS
