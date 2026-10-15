/**************************************************************************/
/* graph_probe.hpp                                                        */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file graph_probe.hpp
 * @brief Declares GraphProbe, a graph a test owns so it can read what its record calls added.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/recorded_graph.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Holds what the compiled plan reports for one recorded task.
         * @see modules/engine/graph_probe/README.md
         */
        struct RecordedTask
        {
            /** @brief The label the task reports under. */
            std::string name;

            /** @brief The operation the task names; a default id names none. */
            SushiRuntime::OperationID operation;

            /** @brief How many recorded tasks the task waits for. */
            std::size_t dependency_count = 0;

            /** @brief The regions the task declared as read, in the order given. */
            std::vector<SushiRuntime::BufferRegion> reads;

            /** @brief The regions the task declared as written, in the order given. */
            std::vector<SushiRuntime::BufferRegion> writes;
        };

        /**
         * @brief Owns a graph, a scratch pool and a recorder over them on the test runtime.
         * @see modules/engine/graph_probe/README.md
         */
        class GraphProbe final
        {
            public:
                /** @brief Builds an empty graph whose recorder records under mode. */
                explicit GraphProbe(ExecutionMode mode = ExecutionMode::Auto)
                    : scratch_pool_(TestEnvironment::runtime()),
                      graph_(TestEnvironment::runtime()),
                      recorder_(graph_.recorder(), scratch_pool_, mode)
                {

                }

                GraphProbe(const GraphProbe&) = delete;
                GraphProbe& operator=(const GraphProbe&) = delete;
                GraphProbe(GraphProbe&&) = delete;
                GraphProbe& operator=(GraphProbe&&) = delete;
                ~GraphProbe() = default;

                /** @brief Returns the recorder a view or a seam records on this graph through. */
                [[nodiscard]] TaskRecorder& recorder() noexcept { return recorder_; }

                /** @brief Returns the pool tasks recorded here lease their scratch from. */
                [[nodiscard]] ScratchPool& scratch_pool() noexcept { return scratch_pool_; }

                /** @brief Returns how many tasks are recorded and have not run. */
                [[nodiscard]] std::size_t recorded_count() const noexcept
                {
                    return graph_.recorded_count();
                }

                /** @brief Returns how many tasks wait on the graph and drops them. */
                [[nodiscard]] std::size_t take_recorded_count()
                {
                    return graph_.discard();
                }

                /**
                 * @brief Returns the recorded tasks that have not run, in record order, building
                 *        the plan of the graph when none is current.
                 * @return The tasks, none when nothing is recorded, or the error that kept the
                 *         graph from compiling.
                 * @see modules/engine/graph_probe/README.md
                 */
                [[nodiscard]] SushiRuntime::Result<std::vector<RecordedTask>> recorded_tasks()
                {
                    if (graph_.is_empty())
                        return std::vector<RecordedTask>{};

                    SushiRuntime::Result<SushiRuntime::PlanView> plan = graph_.plan();
                    if (!plan)
                        return SushiRuntime::make_error(std::move(plan.error()));

                    std::vector<RecordedTask> tasks;
                    tasks.reserve(plan->nodes.size());
                    for (SushiRuntime::PlanNode& node : plan->nodes)
                    {
                        tasks.push_back(RecordedTask{std::move(node.name), node.operation,
                                                     node.predecessor_count,
                                                     std::move(node.reads),
                                                     std::move(node.writes)});
                    }
                    return tasks;
                }

                /**
                 * @brief Returns how many recorded tasks each recorded task waits for, in record
                 *        order.
                 * @return The counts, or the error recorded_tasks() returns.
                 */
                [[nodiscard]] SushiRuntime::Result<std::vector<std::size_t>> dependency_counts()
                {
                    SushiRuntime::Result<std::vector<RecordedTask>> tasks = recorded_tasks();
                    if (!tasks)
                        return SushiRuntime::make_error(std::move(tasks.error()));

                    std::vector<std::size_t> counts;
                    counts.reserve(tasks->size());
                    for (const RecordedTask& task : *tasks)
                        counts.push_back(task.dependency_count);
                    return counts;
                }

                /**
                 * @brief Runs the recorded graph, returns when it has finished and leaves it empty.
                 * @return Nothing, also when no task was recorded, or the error that ended the run.
                 */
                [[nodiscard]] SushiRuntime::Result<void> execute()
                {
                    return graph_.run();
                }

            private:
                /// @brief The pool the recorder hands to operations that lease scratch.
                ScratchPool scratch_pool_;

                /// @brief The graph the test reads.
                RecordedGraph graph_;

                /// @brief The recorder bound to the recorder of graph_ and to scratch_pool_.
                TaskRecorder recorder_;
        };
    } // namespace Test
} // namespace SushiBLAS
