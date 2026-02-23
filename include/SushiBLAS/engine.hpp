/**************************************************************************/
/* engine.hpp                                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                   	  */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/storage.hpp>
#include <SushiBLAS/ops/blas.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/SushiRuntime.h>
#include <SushiBLAS/ops/math/random.hpp>
#include <SushiBLAS/ops/logic/logic.hpp>
#include <SushiBLAS/ops/lapack/linalg.hpp>
#include <SushiBLAS/ops/math/nonlinear.hpp>
#include <SushiBLAS/ops/math/reductions.hpp>
#include <SushiRuntime/graph/task_graph.hpp>
#include <SushiBLAS/ops/math/elementwise.hpp>
#include <SushiBLAS/ops/signal/transforms.hpp>

namespace SushiBLAS 
{
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

            /** @brief Get the default memory layout. */
            inline Core::Layout get_layout() const { return default_layout_; }

            /** @brief Access standard BLAS operations (Levels 1, 2, and 3). */
            inline BLASOps blas() { return BLASOps(*this); }

            /** @brief Logical operations and boolean mask generation. */
            inline LogicOps logic() { return LogicOps(*this); }

            /** @brief High-level linear algebra (LAPACK) solvers. */
            inline LinalgOps linalg() { return LinalgOps(*this); }

            /** @brief Random number generation and tensor initialization. */
            inline RandomOps random() { return RandomOps(*this); }
            
            /** @brief Signal processing and frequency domain transforms (FFT). */
            inline TransformsOps signal() { return TransformsOps(*this); }

            /** @brief Non-linear transformations and activation functions. */
            inline NonLinearOps nonlinear() { return NonLinearOps(*this); }

            /** @brief Access tensor reduction operations (sum, max, etc.). */
            inline ReductionOps reductions() { return ReductionOps(*this); }

            /** @brief Access element-wise arithmetic operations. */
            inline ElementwiseOps elementwise() { return ElementwiseOps(*this); }

            /** @brief Get the underlying SushiRuntime execution context. */
            SushiRuntime::Execution::RuntimeContext& get_context() { return context_; }
            
            /** @brief Get the engine's asynchronous task graph. */
            SushiRuntime::Graph::TaskGraph& get_graph() { return graph_; }

            /** 
             * @brief Execute all queued tasks in the graph. 
             * @return A sycl::event that can be used to synchronize with graph completion.
             */
            sycl::event execute() { return graph_.execute(); }

            /** 
             * @brief Create a new tensor with the specified dimensions.
             * Memory is allocated using the engine's default allocator and layout.
             * @param dims The dimensions of the tensor.
             * @param strat The allocation strategy (Shared, Device, or Host).
             * @return A new Tensor object.
             */
            Tensor create_tensor(std::initializer_list<int64_t> dims, 
                                SushiRuntime::Memory::AllocStrategy strat = SushiRuntime::Memory::AllocStrategy::SHARED);

        private:
            SushiRuntime::Execution::RuntimeContext& context_;
            SushiRuntime::Graph::TaskGraph graph_;
            Core::Layout default_layout_;
    };

} // namespace SushiBLAS
