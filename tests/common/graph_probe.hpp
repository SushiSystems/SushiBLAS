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
 * @brief Declares GraphProbe, a task graph a test owns so it can read what its record calls added.
 * @author Mustafa Garip
 * @see tests/common/README.md
 */

#pragma once

#include <vector>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiRuntime/SushiRuntime.h>
#include <SushiRuntime/graph/node.hpp>
#include <SushiRuntime/graph/task_graph.hpp>
#include <SushiRuntime/execution/execution_engine.hpp>
#include "test_environment.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Owns a task graph, a scratch pool and a recorder over them on the test context.
         * @see tests/common/README.md
         */
        class GraphProbe final
        {
            public:
                /** @brief Builds an empty graph whose recorder records under mode. */
                explicit GraphProbe(ExecutionMode mode = ExecutionMode::Auto)
                    : scratch_pool_(TestEnvironment::get_context()),
                      graph_(TestEnvironment::get_context()),
                      recorder_(graph_, scratch_pool_, mode)
                {

                }

                GraphProbe(const GraphProbe&) = delete;
                GraphProbe& operator=(const GraphProbe&) = delete;
                GraphProbe(GraphProbe&&) = delete;
                GraphProbe& operator=(GraphProbe&&) = delete;
                ~GraphProbe() = default;

                /** @brief Returns the recorder a view or a seam records on this graph through. */
                [[nodiscard]] TaskRecorder& get_recorder() noexcept { return recorder_; }

                /** @brief Returns the pool tasks recorded here lease their scratch from. */
                [[nodiscard]] ScratchPool& get_scratch_pool() noexcept { return scratch_pool_; }

                /** @brief Returns the nodes recorded since the last call and empties the graph. */
                [[nodiscard]] std::vector<SushiRuntime::sushi_ptr<SushiRuntime::Graph::Node>>
                take_nodes()
                {
                    return graph_.take_nodes();
                }

                /** @brief Compiles the recorded graph, runs it and returns when it has finished. */
                void execute()
                {
                    SushiRuntime::Execution::CompiledGraph compiled = graph_.compile();
                    if (compiled.roots.empty())
                        return;

                    TestEnvironment::get_context().get_execution_engine().execute(compiled);
                }

            private:
                /// @brief The pool the recorder hands to operations that lease scratch.
                ScratchPool scratch_pool_;

                /// @brief The graph the test reads.
                SushiRuntime::Graph::TaskGraph graph_;

                /// @brief The recorder bound to graph_ and scratch_pool_.
                TaskRecorder recorder_;
        };
    } // namespace Test
} // namespace SushiBLAS
