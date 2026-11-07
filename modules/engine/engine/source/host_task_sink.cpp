/**************************************************************************/
/* host_task_sink.cpp                                                     */
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
 * @file host_task_sink.cpp
 * @brief Defines HostTaskSink: the kept tasks, and the replay that calls them in record order.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/host_task_sink.hpp>

#include <vector>
#include <utility>
#include <exception>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/pair_run.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>

namespace SushiBLAS
{
    /** @brief Holds the kept tasks in record order. */
    struct HostTaskSink::Tasks
    {
        /** @brief One kept task. */
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
        std::vector<Task> kept;
    };

    HostTaskSink::HostTaskSink() : tasks_(std::make_unique<Tasks>())
    {

    }

    HostTaskSink::~HostTaskSink() = default;

    void HostTaskSink::add_pair(const TaskDescription& description, const OperandSet& operands,
                                ExecutionMode, const std::string&, PairBody body)
    {
        tasks_->kept.push_back(Tasks::Task{description.name, std::move(body), operands.retained()});
    }

    std::size_t HostTaskSink::recorded_count() const noexcept
    {
        return tasks_->kept.size();
    }

    std::optional<HostTaskFailure> HostTaskSink::replay()
    {
        const std::vector<Tasks::Task>& tasks = tasks_->kept;
        const KernelPairDispatcher dispatcher(ExecutionMode::ForceReference);
        const std::vector<sycl::event> no_dependencies;

        for (std::size_t index = 0; index < tasks.size(); ++index)
        {
            try
            {
                PairRun run(dispatcher, no_dependencies);
                tasks[index].body(run);
            }
            catch (const std::exception& error)
            {
                return HostTaskFailure{index, tasks[index].name, error.what()};
            }
            catch (...)
            {
                return HostTaskFailure{index, tasks[index].name,
                                       "the body threw something that is not a std::exception"};
            }
        }
        return std::nullopt;
    }

    std::optional<HostTaskFailure> HostTaskSink::run()
    {
        std::optional<HostTaskFailure> failure = replay();
        tasks_->kept.clear();
        return failure;
    }

    std::size_t HostTaskSink::discard() noexcept
    {
        const std::size_t dropped = tasks_->kept.size();
        tasks_->kept.clear();
        return dropped;
    }
} // namespace SushiBLAS
