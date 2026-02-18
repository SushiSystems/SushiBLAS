/**************************************************************************/
/* SushiBLAS.h                                                            */
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

/**
 * @file SushiBLAS.h
 * @brief Master header for the SushiBLAS library.
 * 
 * Including this file provides access to all public modules of the math library.
 */

// SushiRuntime
#include <SushiRuntime/SushiRuntime.h>

// Core Infrastructure
#include "core/common.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "core/storage.hpp"

namespace SushiBLAS 
{
    /**
     * @brief High-performance BLAS Engine powered by SushiRuntime.
     * 
     * This class acts as the main interface for all mathematical operations.
     * It leverages SushiRuntime's task graph and USM memory management.
     */
    class Engine 
    {
        public:
            /**
             * @brief Construct a new Engine object.
             * @param ctx The SushiRuntime context to use for execution.
             */
            Engine(SushiRuntime::Execution::RuntimeContext& ctx) : context_(ctx) 
            {
                SB_LOG_INFO("SushiBLAS Engine initialized.");
            }

            /**
             * @brief Matrix-Matrix Multiplication (C = alpha*A*B + beta*C).
             * @param A Input tensor A.
             * @param B Input tensor B.
             * @param C Output tensor C.
             */
            void gemm(const Tensor& A, const Tensor& B, Tensor& C) 
            {
                // Placeholder for actual implementation using TaskGraph
                SB_LOG_INFO("GEMM Operation: A({}) x B({}) -> C({})", 
                            (void*)A.data(), (void*)B.data(), (void*)C.data());
            }

            /**
             * @brief Returns the runtime context associated with this engine.
             */
            SushiRuntime::Execution::RuntimeContext& get_context() { return context_; }

            /**
             * @brief Creates a tensor with the specified dimensions and allocation strategy.
             * @param dims Dimensions of the tensor.
             * @param strat Memory allocation strategy (shared, device, host).
             * @return Tensor The created tensor.
             */
            Tensor create_tensor(std::initializer_list<int64_t> dims, 
                                     SushiRuntime::Memory::AllocStrategy strat = SushiRuntime::Memory::AllocStrategy::SHARED)
            {
                size_t elements = 1;
                for (auto d : dims) elements *= d;
                
                auto storage = SushiRuntime::make_sushi<Storage>(context_.get_allocator(), elements * sizeof(float), strat);
                return Tensor(storage, dims);
            }

        private:
            SushiRuntime::Execution::RuntimeContext& context_;
    };
} // namespace SushiBLAS
