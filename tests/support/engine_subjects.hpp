/**************************************************************************/
/* engine_subjects.hpp                                                    */
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
 * @file engine_subjects.hpp
 * @brief Declares the subjects an Engine contract runs on: a host engine and an engine on the shared runtime.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <optional>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine_fixture/host_recording_fixture.hpp>
#include <SushiBLAS/engine_fixture/recording_fixture.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds a host engine and a recorder over the host sink its fixture owns. */
        class HostEngineSubject
        {
            public:
                /** @brief Returns the engine under test. */
                [[nodiscard]] Engine& engine() noexcept { return engine_; }

                /** @brief Returns the recorder over the fixture's host sink. */
                [[nodiscard]] TaskRecorder& recorder() noexcept { return recording_.recorder(); }

                /**
                 * @brief Runs the recorded tasks.
                 * @return Empty when all ran, else the name of the failed task and its message.
                 */
                [[nodiscard]] std::string run()
                {
                    const std::optional<HostTaskFailure> failure = recording_.run();
                    if (!failure.has_value())
                        return std::string();
                    return failure->name + ": " + failure->message;
                }

            private:
                /// @brief The engine under test.
                Engine engine_{Engine::Host{}};

                /// @brief The recorder and run over engine_.
                HostRecordingFixture recording_{engine_};
        };

        /** @brief Holds an engine on the runtime the test program made, set to ForceReference, and a graph recorder. */
        class RuntimeEngineSubject
        {
            public:
                /** @brief Builds the engine and sets it to ForceReference before the recorder reads its mode. */
                RuntimeEngineSubject()
                {
                    engine_.set_mode(ExecutionMode::ForceReference);
                    recording_.emplace(engine_);
                }

                /** @brief Returns the engine under test. */
                [[nodiscard]] Engine& engine() noexcept { return engine_; }

                /** @brief Returns the recorder over the engine graph. */
                [[nodiscard]] TaskRecorder& recorder() noexcept { return recording_->recorder(); }

                /**
                 * @brief Runs the recorded graph.
                 * @return Empty when it ran, else the runtime message.
                 */
                [[nodiscard]] std::string run()
                {
                    const SushiRuntime::Result<void> ran = recording_->run();
                    if (ran)
                        return std::string();
                    return ran.error().message;
                }

            private:
                /// @brief The engine under test.
                Engine engine_{TestEnvironment::runtime()};

                /// @brief The graph and recorder over engine_.
                std::optional<RecordingFixture> recording_;
        };
    } // namespace Test
} // namespace SushiBLAS
