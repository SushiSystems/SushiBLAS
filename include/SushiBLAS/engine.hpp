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
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/SushiRuntime.h>
#include <SushiRuntime/graph/task_graph.hpp>

// Operations Interfaces
#include <SushiBLAS/ops/blas.hpp>
#include <SushiBLAS/ops/math/activations.hpp>
#include <SushiBLAS/ops/math/reductions.hpp>
#include <SushiBLAS/ops/math/elementwise.hpp>


namespace SushiBLAS 
{
    /**
     * @brief High-performance math engine.
     */
    class Engine 
    {
        public:
            Engine(SushiRuntime::Execution::RuntimeContext& ctx, 
                   Core::Layout layout = Core::Layout::ROW_MAJOR);

            /** @brief Access default layout. */
            inline Core::Layout get_layout() const { return default_layout_; }

            /** @brief Standard BLAS operations. */
            inline BLASOps blas() { return BLASOps(*this); }

            /** @brief NN activation functions. */
            inline ActivationOps activations() { return ActivationOps(*this); }

            /** @brief Tensor reduction operations. */
            inline ReductionOps reductions() { return ReductionOps(*this); }

            /** @brief Element-wise operations. */
            inline ElementwiseOps elementwise() { return ElementwiseOps(*this); }

            /** @brief Returns source runtime context. */
            SushiRuntime::Execution::RuntimeContext& get_context() { return context_; }
            
            /** @brief Returns the engine's task graph. */
            SushiRuntime::Graph::TaskGraph& get_graph() { return graph_; }

            /** @brief Submits the graph for execution and waits for completion. */
            sycl::event execute() { return graph_.execute(); }

            /** @brief Creates a tensor. */
            Tensor create_tensor(std::initializer_list<int64_t> dims, 
                                SushiRuntime::Memory::AllocStrategy strat = SushiRuntime::Memory::AllocStrategy::SHARED);

        private:
            SushiRuntime::Execution::RuntimeContext& context_;
            SushiRuntime::Graph::TaskGraph graph_;
            Core::Layout default_layout_;
    };

} // namespace SushiBLAS
