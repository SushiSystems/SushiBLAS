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
 * @brief Declares the Engine, which owns one task graph, the bricks that record on it, and runs it.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor_factory.hpp>
#include <SushiBLAS/engine/rng_stream.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiRuntime/SushiRuntime.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/graph/task_graph.hpp>

namespace SushiBLAS
{
    /**
     * @class Engine
     * @brief Owns one task graph with its scratch pool, recorder, tensor factory and random
     *        stream, and runs the graph.
     * @see include/SushiBLAS/README.md
     */
    class Engine final
    {
        public:
            /**
             * @brief Builds an engine on a runtime context.
             * @param context Must outlive the engine and every tensor it creates.
             * @param layout The memory layout of tensors this engine creates.
             */
            explicit Engine(SushiRuntime::Execution::RuntimeContext& context,
                            Core::Layout layout = Core::Layout::ROW_MAJOR);

            Engine(const Engine&) = delete;
            Engine& operator=(const Engine&) = delete;
            Engine(Engine&&) = delete;
            Engine& operator=(Engine&&) = delete;
            ~Engine() = default;

            /** @brief Returns the recorder every operation view of this engine records through. */
            [[nodiscard]] TaskRecorder& get_recorder() noexcept { return recorder_; }

            /** @brief Returns the factory create_tensor forwards to. */
            [[nodiscard]] TensorFactory& get_tensor_factory() noexcept { return tensor_factory_; }

            /**
             * @brief Returns the random stream every RandomOps view of this engine draws from.
             * @see include/SushiBLAS/engine/README.md
             */
            [[nodiscard]] RngStream& get_rng() noexcept { return rng_; }

            /**
             * @brief Returns the pool tasks recorded through this engine lease device scratch from.
             * @see include/SushiBLAS/engine/README.md
             */
            [[nodiscard]] ScratchPool& get_scratch_pool() noexcept { return scratch_pool_; }

            /** @brief Returns the runtime context the engine was built on. */
            [[nodiscard]] SushiRuntime::Execution::RuntimeContext& get_context() noexcept
            {
                return context_;
            }

            /** @brief Sets the execution mode of every operation recorded from now on. */
            void set_mode(ExecutionMode mode) noexcept { recorder_.set_mode(mode); }

            /** @brief Returns the execution mode operations are recorded under. */
            [[nodiscard]] ExecutionMode get_mode() const noexcept { return recorder_.get_mode(); }

            /** @brief Returns the memory layout of tensors this engine creates. */
            [[nodiscard]] Core::Layout get_layout() const noexcept
            {
                return tensor_factory_.get_layout();
            }

            /**
             * @brief Compiles the graph recorded through get_recorder(), runs it and returns when
             *        every task has finished.
             * @pre No thread is recording on this engine.
             */
            void execute();

            /**
             * @brief Creates a tensor from a brace list of dimensions.
             * @param device_index Device queue that owns the allocation.
             * @return A tensor without storage when a dimension is 0.
             * @throws std::runtime_error as TensorFactory::create states.
             */
            [[nodiscard]] Tensor create_tensor(
                std::initializer_list<int64_t> dims,
                Core::DataType dtype = Core::DataType::FLOAT32,
                SushiRuntime::API::Residency residency = SushiRuntime::API::Residency::Shared,
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
                Core::DataType dtype = Core::DataType::FLOAT32,
                SushiRuntime::API::Residency residency = SushiRuntime::API::Residency::Shared,
                std::size_t device_index = 0)
            {
                return tensor_factory_.create(dims, dtype, residency, device_index);
            }

        private:
            /// @brief The runtime context that allocates and executes.
            SushiRuntime::Execution::RuntimeContext& context_;

            /// @brief The pool recorded tasks lease scratch from; see include/SushiBLAS/README.md.
            ScratchPool scratch_pool_;

            /// @brief The graph recorded tasks wait in until execute(); no accessor reaches it.
            SushiRuntime::Graph::TaskGraph graph_;

            /// @brief The recorder bound to graph_ and scratch_pool_.
            TaskRecorder recorder_;

            /// @brief The factory behind create_tensor.
            TensorFactory tensor_factory_;

            /// @brief The random stream of this engine.
            RngStream rng_;
    };
} // namespace SushiBLAS
