/**************************************************************************/
/* benchmark_graph.hpp                                                    */
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
 * @file benchmark_graph.hpp
 * @brief Declares BenchmarkGraph, the graph and recorder a benchmark program owns.
 * @author Mustafa Garip
 * @see modules/operations/benchmark_harness/README.md
 */

#pragma once

#include <utility>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS::Bench
{
    /// @brief The execution mode a BenchmarkGraph records under unless a benchmark sets another.
    constexpr ExecutionMode K_DEFAULT_MODE = ExecutionMode::Auto;

    /**
     * @brief Owns a runtime graph and a TaskRecorder over it, so a benchmark records and runs
     *        on a graph of its own.
     *
     * It carries no test framework, which is why a benchmark program does not use
     * Test::RecordingFixture.
     * @see modules/operations/benchmark_harness/README.md
     */
    class BenchmarkGraph final
    {
        public:
            /**
             * @brief Builds an empty graph on the engine's runtime and a recorder over it that
             *        borrows the engine's scratch pool.
             * @param engine Must outlive this object.
             * @param mode The execution mode the recorder records under.
             */
            explicit BenchmarkGraph(Engine& engine, ExecutionMode mode = K_DEFAULT_MODE)
                : graph_(engine.runtime().graph()),
                  recorder_(graph_.recorder(), engine.scratch_pool(), mode)
            {

            }

            BenchmarkGraph(const BenchmarkGraph&) = delete;
            BenchmarkGraph& operator=(const BenchmarkGraph&) = delete;
            BenchmarkGraph(BenchmarkGraph&&) = delete;
            BenchmarkGraph& operator=(BenchmarkGraph&&) = delete;
            ~BenchmarkGraph() = default;

            /** @brief Returns the recorder an operation view records on this graph through. */
            [[nodiscard]] TaskRecorder& recorder() noexcept { return recorder_; }

            /** @brief Returns the recorder an operation view records on this graph through. */
            [[nodiscard]] const TaskRecorder& recorder() const noexcept { return recorder_; }

            /**
             * @brief Runs the recorded tasks, returns when every one has finished, and leaves
             *        the graph empty.
             * @return Nothing, or the error that ended the run.
             * @pre No thread is recording on recorder().
             */
            [[nodiscard]] SushiRuntime::Result<void> run()
            {
                SushiRuntime::Result<SushiRuntime::RunReport*> ran = graph_.run(report_);
                recorder_.run_exclusive([this]() { graph_.clear(); });
                if (ran)
                    return {};
                return SushiRuntime::make_error(std::move(ran.error()));
            }

        private:
            /// @brief The graph recorded tasks wait in.
            SushiRuntime::Graph graph_;

            /// @brief The recorder bound to the recorder of graph_ and to the engine's scratch pool.
            TaskRecorder recorder_;

            /// @brief The report every run writes into, so a run allocates none.
            SushiRuntime::RunReport report_;
    };
} // namespace SushiBLAS::Bench
