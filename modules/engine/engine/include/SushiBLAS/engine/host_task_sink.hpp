/**************************************************************************/
/* host_task_sink.hpp                                                     */
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
 * @file host_task_sink.hpp
 * @brief Declares HostTaskSink, the TaskSink that keeps bodies in record order and runs them on
 *        the host under ForceReference.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <memory>
#include <cstddef>
#include <optional>
#include <SushiBLAS/engine/task_sink.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>

namespace SushiBLAS
{
    /** @brief Names the task that failed in HostTaskSink::run() and why. */
    struct HostTaskFailure
    {
        /// @brief The place of the task in record order, from 0.
        std::size_t index;

        /// @brief The name of the task's description.
        std::string name;

        /// @brief The what() of the exception the body threw, or a fixed text for one that is not a std::exception.
        std::string message;
    };

    /**
     * @brief Keeps recorded bodies in record order and runs them on the host, as often as its
     *        owner replays them: replay() keeps them, discard() drops them, run() does both.
     *
     * A body gets a dispatcher fixed to ForceReference that holds no queue. Not thread-safe
     * beyond TaskRecorder's own serialisation of add_pair().
     * @see modules/engine/engine/README.md
     */
    class HostTaskSink final : public TaskSink
    {
        public:
            /** @brief Holds no task. */
            HostTaskSink();
            ~HostTaskSink() override;

            /** @brief Reports true for ForceReference only. */
            [[nodiscard]] bool admits(ExecutionMode mode) const noexcept override
            {
                return mode == ExecutionMode::ForceReference;
            }

            /**
             * @brief Keeps the body after the ones recorded before, with the operand storages it reads.
             * @pre The mode is ForceReference; TaskRecorder refuses any other before it calls.
             */
            void add_pair(const TaskDescription& description, const OperandSet& operands,
                          ExecutionMode mode, const std::string& origin, PairBody body) override;

            /** @brief Returns how many tasks are kept and have not run. */
            [[nodiscard]] std::size_t recorded_count() const noexcept;

            /**
             * @brief Runs the kept bodies in record order and keeps them for the next replay.
             *
             * The first body that throws stops the replay; the bodies after it do not run, and
             * every body stays kept.
             * @return Nothing when every body ran, else the first failure.
             */
            [[nodiscard]] std::optional<HostTaskFailure> replay();

            /**
             * @brief Replays the kept bodies once, then drops them all, whether or not one failed.
             * @return What replay() returned.
             */
            [[nodiscard]] std::optional<HostTaskFailure> run();

            /** @brief Drops every kept task without running it and returns how many were dropped. */
            std::size_t discard() noexcept;

        private:
            /// @brief The kept tasks, defined in the source.
            struct Tasks;

            /// @brief The kept tasks in record order; never null.
            std::unique_ptr<Tasks> tasks_;
    };
} // namespace SushiBLAS
