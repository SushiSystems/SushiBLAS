/**************************************************************************/
/* host_graph_probe.hpp                                                   */
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
 * @file host_graph_probe.hpp
 * @brief Declares HostGraphProbe, a graph of host tasks a test owns so it can read the
 *        dependencies its record calls declared without a runtime.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <utility>
#include <SushiBLAS/engine/host_task_sink.hpp>
#include <SushiBLAS/engine/host_scratch_source.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Owns a HostTaskSink, a host scratch pool and a recorder over them, and reports
         *        each recorded task with the tasks it waits for.
         *
         * It makes no runtime and no queue, so a test over it is Small_. A tensor needs host
         * storage only: the probe reads the regions a task declared, not the memory behind them.
         * @see modules/engine/graph_probe/README.md
         */
        class HostGraphProbe final
        {
            public:
                /** @brief Builds an empty graph whose recorder records under ForceReference. */
                HostGraphProbe() : pool_(source_), recorder_(sink_, &pool_) {}

                HostGraphProbe(const HostGraphProbe&) = delete;
                HostGraphProbe& operator=(const HostGraphProbe&) = delete;
                HostGraphProbe(HostGraphProbe&&) = delete;
                HostGraphProbe& operator=(HostGraphProbe&&) = delete;
                ~HostGraphProbe() = default;

                /** @brief Returns the recorder a view or a seam records on this graph through. */
                [[nodiscard]] TaskRecorder& recorder() noexcept { return recorder_; }

                /** @brief Returns how many tasks are recorded and have not run. */
                [[nodiscard]] std::size_t recorded_count() const noexcept
                {
                    return sink_.recorded_count();
                }

                /**
                 * @brief Returns the recorded tasks in record order, each with the number of
                 *        earlier tasks whose declared regions conflict with its own.
                 * @see modules/engine/graph_probe/README.md
                 */
                [[nodiscard]] std::vector<RecordedTask> recorded_tasks() const
                {
                    const std::vector<HostTaskRecord> records = sink_.records();

                    std::vector<RecordedTask> tasks;
                    tasks.reserve(records.size());
                    for (std::size_t later = 0; later < records.size(); ++later)
                    {
                        std::size_t waits = 0;
                        for (std::size_t earlier = 0; earlier < later; ++earlier)
                            if (conflicts(records[earlier], records[later]))
                                ++waits;

                        const HostTaskRecord& record = records[later];
                        tasks.push_back(RecordedTask{record.name, record.operation, waits,
                                                     record.reads, record.writes, std::nullopt,
                                                     record.origin});
                    }
                    return tasks;
                }

                /** @brief Returns how many earlier tasks each recorded task waits for. */
                [[nodiscard]] std::vector<std::size_t> dependency_counts() const
                {
                    std::vector<std::size_t> counts;
                    for (const RecordedTask& task : recorded_tasks())
                        counts.push_back(task.dependency_count);
                    return counts;
                }

            private:
                /** @brief Returns whether any region of the first list overlaps one of the second. */
                [[nodiscard]] static bool overlap(const std::vector<SushiRuntime::BufferRegion>& first,
                                                  const std::vector<SushiRuntime::BufferRegion>& second)
                {
                    for (const SushiRuntime::BufferRegion& a : first)
                        for (const SushiRuntime::BufferRegion& b : second)
                            if (a.overlaps(b))
                                return true;
                    return false;
                }

                /** @brief Returns whether the later task must wait for the earlier one. */
                [[nodiscard]] static bool conflicts(const HostTaskRecord& earlier,
                                                    const HostTaskRecord& later)
                {
                    return overlap(earlier.writes, later.reads) ||
                           overlap(earlier.writes, later.writes) ||
                           overlap(earlier.reads, later.writes);
                }

                /// @brief The source of the blocks the pool lends; declared first, used by pool_.
                HostScratchSource source_;

                /// @brief The pool the recorder hands to operations that lease scratch.
                ScratchPool pool_;

                /// @brief The tasks, in record order.
                HostTaskSink sink_;

                /// @brief The recorder over sink_ and pool_.
                TaskRecorder recorder_;
        };
    } // namespace Test
} // namespace SushiBLAS
