/**************************************************************************/
/* engine.hpp                                                             */
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
 * @file engine.hpp
 * @brief Declares the Engine, which owns one graph, the bricks that record on it, and runs it.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/engine/rng_stream.hpp>
#include <SushiBLAS/engine/recorded_graph.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/span.hpp>

namespace SushiBLAS
{
    /**
     * @class Engine
     * @brief Owns one graph with its scratch pool, recorder, tensor factory and random
     *        stream, and runs the graph.
     * @see modules/README.md
     */
    class Engine final
    {
        public:
            /**
             * @brief Builds an engine on a runtime.
             * @param runtime Must outlive the engine.
             * @param layout The memory layout of tensors this engine creates.
             */
            explicit Engine(SushiRuntime::Runtime& runtime,
                            Core::Layout layout = Core::Layout::RowMajor);

            Engine(const Engine&) = delete;
            Engine& operator=(const Engine&) = delete;
            Engine(Engine&&) = delete;
            Engine& operator=(Engine&&) = delete;
            ~Engine() = default;

            /** @brief Returns the recorder every operation view of this engine records through. */
            [[nodiscard]] TaskRecorder& recorder() noexcept { return recorder_; }

            /** @brief Returns the factory create_tensor forwards to. */
            [[nodiscard]] TensorFactory& tensor_factory() noexcept { return tensor_factory_; }

            /**
             * @brief Returns the random stream every RandomOps view of this engine draws from.
             * @see modules/engine/engine/README.md
             */
            [[nodiscard]] RNGStream& rng() noexcept { return rng_; }

            /**
             * @brief Returns the pool tasks recorded through this engine lease device scratch from.
             * @see modules/engine/engine/README.md
             */
            [[nodiscard]] ScratchPool& scratch_pool() noexcept { return scratch_pool_; }

            /** @brief Returns the runtime the engine was built on. */
            [[nodiscard]] SushiRuntime::Runtime& runtime() noexcept
            {
                return runtime_;
            }

            /**
             * @brief Sets the execution mode of every operation recorded from now on.
             * @pre No thread is recording on this engine.
             */
            void set_mode(ExecutionMode mode) noexcept { recorder_.set_mode(mode); }

            /** @brief Returns the execution mode operations are recorded under. */
            [[nodiscard]] ExecutionMode mode() const noexcept { return recorder_.mode(); }

            /** @brief Returns the memory layout of tensors this engine creates. */
            [[nodiscard]] Core::Layout layout() const noexcept
            {
                return tensor_factory_.layout();
            }

            /**
             * @brief Runs the graph recorded through recorder(), returns when every task has
             *        finished, and leaves the graph empty.
             * @return Nothing, also when no task was recorded, or the error that ended the run.
             *         After InvalidState for a run in flight the recorded tasks are kept.
             * @pre No thread is recording on this engine.
             * @see modules/README.md
             */
            [[nodiscard]] SushiRuntime::Result<void> execute();

            /**
             * @brief Drops every task recorded through recorder() that has not run, with the
             *        operand storage and the scratch those tasks hold; the engine stays usable.
             * @pre No thread is running execute() on this engine.
             * @see modules/README.md
             */
            void discard_recorded();

            /**
             * @brief Creates a tensor from a brace list of dimensions.
             * @param device_index Device queue that owns the allocation.
             * @return A tensor without storage when a dimension is 0.
             * @throws std::runtime_error as TensorFactory::create states.
             */
            [[nodiscard]] Tensor create_tensor(
                std::initializer_list<int64_t> dims,
                Core::DataType dtype = Core::DataType::Float32,
                SushiRuntime::Residency residency = SushiRuntime::Residency::Shared,
                std::size_t device_index = 0)
            {
                return tensor_factory_.create(
                    SushiRuntime::span<const int64_t>(dims.begin(), dims.end()), dtype, residency,
                    device_index);
            }

            /**
             * @brief Creates a tensor whose dimensions are known only at run time.
             * @param device_index Device queue that owns the allocation.
             * @return A tensor without storage when a dimension is 0.
             * @throws std::runtime_error as TensorFactory::create states.
             */
            [[nodiscard]] Tensor create_tensor(
                SushiRuntime::span<const int64_t> dims,
                Core::DataType dtype = Core::DataType::Float32,
                SushiRuntime::Residency residency = SushiRuntime::Residency::Shared,
                std::size_t device_index = 0)
            {
                return tensor_factory_.create(dims, dtype, residency, device_index);
            }

        private:
            /// @brief The runtime that allocates and executes.
            SushiRuntime::Runtime& runtime_;

            /// @brief The pool recorded tasks lease scratch from; see modules/README.md.
            ScratchPool scratch_pool_;

            /// @brief The graph recorded tasks wait in until execute(); no accessor reaches it.
            RecordedGraph graph_;

            /// @brief The recorder bound to the recorder of graph_ and to scratch_pool_.
            TaskRecorder recorder_;

            /// @brief The factory behind create_tensor.
            TensorFactory tensor_factory_;

            /// @brief The random stream of this engine.
            RNGStream rng_;
    };
} // namespace SushiBLAS
