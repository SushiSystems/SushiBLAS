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
 * @brief Declares the Engine, which records operations on a task graph and executes them.
 * @author Mustafa Garip
 */

#pragma once

#include <atomic>
#include <SushiBLAS/io.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/storage.hpp>
#include <SushiBLAS/engine/blas.hpp>
#include <SushiBLAS/engine/spatial.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/SushiRuntime.h>
#include <SushiBLAS/engine/logic/logic.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiBLAS/engine/math/nonlinear.hpp>
#include <SushiRuntime/graph/task_graph.hpp>
#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/elementwise.hpp>


namespace SushiBLAS 
{
    /**
     * @enum ExecutionMode
     * @brief Defines how kernels are dispatched.
     */
    enum class ExecutionMode 
    {
        AUTO,              // Heuristic-based selection
        FORCE_SYCL,        // Always use SYCL/MKL
        FORCE_REFERENCE,   // Always use Reference (CPU)
        VERIFY             // Run both and compare (Debug mode)
    };

    /**
     * @class Engine
     * @brief The core engine for high-performance mathematical operations.
     * 
     * The Engine manages the execution context, task graph, and memory allocation 
     * for all SushiBLAS operations. It acts as the primary entry point for 
     * performing BLAS and other mathematical computations.
     */
    class Engine 
    {
        public:
            /**
             * @brief Construct a new Engine.
             * @param ctx The SushiRuntime execution context.
             * @param layout The default memory layout for tensors created by this engine.
             */
            Engine(SushiRuntime::Execution::RuntimeContext& ctx, 
                   Core::Layout layout = Core::Layout::ROW_MAJOR);

            /** 
             * @brief Set the execution mode. 
             * @param mode The desired execution mode.
             */
            void set_mode(ExecutionMode mode) { mode_ = mode; }

            /** 
             * @brief Get the current execution mode. 
             * @return The current execution mode.
             */
            ExecutionMode get_mode() const { return mode_; }

            /** 
             * @brief Get the default memory layout. 
             * @return The current default layout (Row-Major or Column-Major).
             */
            inline Core::Layout get_layout() const { return default_layout_; }

            /** 
             * @brief Access standard BLAS operations (Levels 1, 2, and 3). 
             * @return A BLASOps object providing access to BLAS routines.
             */
            inline BLASOps blas() { return BLASOps(*this); }

            /** 
             * @brief Input/Output operations (Persistence & Print). 
             * @return An IO object providing access to save, load and print routines.
             */
            inline IO io() { return IO(*this); }

            /** 
             * @brief Logical operations and boolean mask generation. 
             * @return A LogicOps object providing access to logical kernels.
             */
            inline LogicOps logic() { return LogicOps(*this); }

            /**
             * @brief Random number generation and tensor initialization.
             * @return A RandomOps object providing access to PRNG kernels.
             */
            inline RandomOps random() { return RandomOps(*this); }

            /**
             * @brief Non-linear transformations and activation functions. 
             * @return A NonLinearOps object providing access to non-linear kernels.
             */
            inline NonLinearOps nonlinear() { return NonLinearOps(*this); }

            /** 
             * @brief Access tensor reduction operations (sum, max, etc.). 
             * @return A ReductionOps object providing access to reduction kernels.
             */
            inline ReductionOps reductions() { return ReductionOps(*this); }

            /**
             * @brief Access windowed and strided movement over an N-D grid.
             * @return A SpatialOps object providing access to the spatial kernels.
             */
            inline SpatialOps spatial() { return SpatialOps(*this); }

            /** 
             * @brief Access element-wise arithmetic operations. 
             * @return An ElementwiseOps object providing access to arithmetic kernels.
             */
            inline ElementwiseOps elementwise() { return ElementwiseOps(*this); }

            /** 
             * @brief Get the underlying SushiRuntime execution context. 
             * @return Reference to the RuntimeContext.
             */
            SushiRuntime::Execution::RuntimeContext& get_context() { return context_; }
            
            /** 
             * @brief Get the engine's asynchronous task graph. 
             * @return Reference to the TaskGraph.
             */
            SushiRuntime::Graph::TaskGraph& get_graph() { return graph_; }

            /**
             * @brief Execute all queued tasks in the graph.
             * @return A sycl::event that can be used to synchronize with graph completion.
             */
            sycl::event execute()
            {
                auto compiled = graph_.compile();
                // An empty graph returns early; the execution engine throws on one; see include/SushiBLAS/README.md.
                if (compiled.roots.empty()) return sycl::event();
                return context_.get_execution_engine().execute(compiled);
            }

            /** 
             * @brief Create a new FLOAT32 tensor with the specified dimensions.
             * @param dims The dimensions of the tensor.
             * @param strat The allocation strategy (Shared, Device, or Host).
             * @return A new FLOAT32 Tensor object.
             */
            Tensor create_tensor(std::initializer_list<int64_t> dims,
                                 const SushiRuntime::Memory::IAllocatorStrategy& strat = SushiRuntime::Memory::Strategies::Shared::instance());

            /** 
             * @brief Create a new tensor with the specified dimensions and data type.
             * @param dims The dimensions of the tensor.
             * @param dtype The data type of the tensor (e.g., FLOAT64, COMPLEX32).
             * @param strat The allocation strategy (Shared, Device, or Host).
             * @return A new Tensor object.
             */
            Tensor create_tensor(std::initializer_list<int64_t> dims,
                                 Core::DataType dtype,
                                 const SushiRuntime::Memory::IAllocatorStrategy& strat = SushiRuntime::Memory::Strategies::Shared::instance());

            /**
             * @brief Sets the random seed and rewinds the counter offset to zero.
             * @see include/SushiBLAS/README.md
             */
            void set_seed(uint64_t s)
            {
                seed_ = s;
                rng_offset_.store(0, std::memory_order_relaxed);
            }

            /**
             * @brief Get the current seed value.
             * @return The seed value.
             */
            uint64_t get_seed() const { return seed_; }

            /**
             * @brief Returns the current RNG offset and increments it.
             * @see include/SushiBLAS/README.md
             */
            uint64_t get_and_increment_rng_offset() { return rng_offset_++; }

        private:
            SushiRuntime::Execution::RuntimeContext& context_;
            SushiRuntime::Graph::TaskGraph graph_;
            Core::Layout default_layout_;
            ExecutionMode mode_ = ExecutionMode::AUTO;
            uint64_t seed_ = 0;
            std::atomic<uint64_t> rng_offset_{0};
    };

} // namespace SushiBLAS
