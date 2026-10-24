/**************************************************************************/
/* recording_fixture.hpp                                                  */
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
 * @file recording_fixture.hpp
 * @brief Declares RecordingFixture, a RecordingGraph a test owns, with its GoogleTest helpers.
 * @author Mustafa Garip
 * @see modules/engine/engine_fixture/README.md
 */

#pragma once

#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/recording_graph/recording_graph.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Holds a RecordingGraph for a test and adds the GoogleTest assertion on a run.
         * @see modules/engine/engine_fixture/README.md
         */
        class RecordingFixture final
        {
            public:
                /**
                 * @brief Builds an empty graph on a runtime and a recorder over it.
                 * @param runtime Must outlive the fixture.
                 * @param scratch_pool Lends operation scratch; it must outlive the fixture.
                 * @param mode The execution mode the recorder records under.
                 */
                RecordingFixture(SushiRuntime::Runtime& runtime, ScratchPool& scratch_pool,
                                 ExecutionMode mode = ExecutionMode::Auto)
                    : graph_(runtime, scratch_pool, mode)
                {

                }

                /**
                 * @brief Builds an empty graph on an engine's runtime and a recorder over it that
                 *        borrows the engine's scratch pool and execution mode.
                 * @param engine Must outlive the fixture.
                 */
                explicit RecordingFixture(Engine& engine)
                    : graph_(engine)
                {

                }

                RecordingFixture(const RecordingFixture&) = delete;
                RecordingFixture& operator=(const RecordingFixture&) = delete;
                RecordingFixture(RecordingFixture&&) = delete;
                RecordingFixture& operator=(RecordingFixture&&) = delete;
                ~RecordingFixture() = default;

                /** @brief Returns the recorder a view or a seam records on this graph through. */
                [[nodiscard]] TaskRecorder& recorder() noexcept { return graph_.recorder(); }

                /** @brief Returns how many tasks are recorded and have not run. */
                [[nodiscard]] std::size_t recorded_count() const noexcept
                {
                    return graph_.recorded_count();
                }

                /** @brief Reports whether a run would have nothing to do and nothing to report. */
                [[nodiscard]] bool is_empty() const noexcept { return graph_.is_empty(); }

                /** @brief Returns what the plan orders, as RecordingGraph::plan() does. */
                [[nodiscard]] SushiRuntime::Result<SushiRuntime::PlanView> plan()
                {
                    return graph_.plan();
                }

                /** @brief Runs the recorded graph as RecordingGraph::run() does. */
                [[nodiscard]] SushiRuntime::Result<void> run() { return graph_.run(); }

                /**
                 * @brief Runs the recorded graph as run() does and reports whether it succeeded.
                 * @return Success, or a failure that carries the runtime's message.
                 */
                [[nodiscard]] ::testing::AssertionResult run_succeeds()
                {
                    const SushiRuntime::Result<void> ran = run();
                    if (ran)
                        return ::testing::AssertionSuccess();
                    return ::testing::AssertionFailure()
                           << "the graph did not run: " << ran.error().message;
                }

                /** @brief Drops every recorded task as RecordingGraph::discard() does. */
                std::size_t discard() { return graph_.discard(); }

            private:
                /// @brief The graph and recorder this fixture holds.
                RecordingGraph graph_;
        };
    } // namespace Test
} // namespace SushiBLAS
