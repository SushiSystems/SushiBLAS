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
 * @brief Declares HostTaskSink, the TaskSink that keeps bodies and runs them on the host under ForceReference.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <utility>
#include <optional>
#include <exception>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/task_sink.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>

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
     * @brief Keeps the bodies of recorded tasks in record order and runs them on the host.
     *
     * It makes no queue and holds no runtime. A body gets a dispatcher fixed to ForceReference
     * that holds no queue, so its reference half runs and queue() throws. Not thread-safe
     * beyond TaskRecorder's own serialisation of add_pair().
     * @see modules/engine/engine/README.md
     */
    class HostTaskSink final : public TaskSink
    {
        public:
            /** @brief Holds no task. */
            HostTaskSink() = default;

            /** @brief Reports true for ForceReference only. */
            [[nodiscard]] bool admits(ExecutionMode mode) const noexcept override
            {
                return mode == ExecutionMode::ForceReference;
            }

            /** @brief Reports false: the dispatcher of a host body holds no queue. */
            [[nodiscard]] bool provides_queue() const noexcept override { return false; }

            /** @brief Returns null: the host form has no graph. */
            [[nodiscard]] GraphTaskSink* graph() noexcept override { return nullptr; }

            /**
             * @brief Keeps the body after the ones recorded before, with the operand storages it reads.
             * @pre The mode is ForceReference; TaskRecorder refuses any other before it calls.
             */
            void add_pair(const TaskDescription& description, const OperandSet& operands,
                          ExecutionMode, const std::string&, PairBody body) override
            {
                tasks_.push_back(Task{description.name, std::move(body), operands.retained()});
            }

            /** @brief Returns how many tasks are kept and have not run. */
            [[nodiscard]] std::size_t recorded_count() const noexcept { return tasks_.size(); }

            /**
             * @brief Runs the kept bodies in record order, then drops them all.
             *
             * The first body that throws stops the run; the bodies after it do not run and are dropped.
             * @return Nothing when every body ran, else the first failure.
             */
            [[nodiscard]] std::optional<HostTaskFailure> run()
            {
                std::vector<Task> tasks = std::move(tasks_);
                tasks_.clear();
                const KernelPairDispatcher dispatcher(ExecutionMode::ForceReference);
                const std::vector<sycl::event> no_dependencies;

                for (std::size_t index = 0; index < tasks.size(); ++index)
                {
                    try
                    {
                        (void)tasks[index].body(dispatcher, no_dependencies);
                    }
                    catch (const std::exception& error)
                    {
                        return HostTaskFailure{index, tasks[index].name, error.what()};
                    }
                    catch (...)
                    {
                        return HostTaskFailure{
                            index, tasks[index].name,
                            "the body threw something that is not a std::exception"};
                    }
                }
                return std::nullopt;
            }

            /** @brief Drops every kept task without running it and returns how many were dropped. */
            std::size_t discard() noexcept
            {
                const std::size_t dropped = tasks_.size();
                tasks_.clear();
                return dropped;
            }

        private:
            /// @brief One kept task.
            struct Task
            {
                /// @brief The name of the task's description, copied.
                std::string name;

                /// @brief The recorded body.
                PairBody body;

                /// @brief The storages that stay allocated until the task is dropped.
                Internal::StorageKeepAlive retained;
            };

            /// @brief The kept tasks in record order.
            std::vector<Task> tasks_;
    };
} // namespace SushiBLAS
