/**************************************************************************/
/* engine.cpp                                                             */
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

#include <SushiBLAS/engine.hpp>

namespace SushiBLAS 
{
    Engine::Engine(SushiRuntime::Execution::RuntimeContext& ctx, Core::Layout layout) 
        : context_(ctx), graph_(ctx), default_layout_(layout) 
    {
        SB_LOG_INFO("SushiBLAS Engine initialized with {} layout.", 
                    layout == Core::Layout::ROW_MAJOR ? "Row-Major" : "Column-Major");
    }

    Tensor Engine::create_tensor(std::initializer_list<int64_t> dims, 
                                 SushiRuntime::Memory::AllocStrategy strat)
    {
        return create_tensor(dims, Core::DataType::FLOAT32, strat);
    }

    Tensor Engine::create_tensor(std::initializer_list<int64_t> dims, 
                                 Core::DataType dtype,
                                 SushiRuntime::Memory::AllocStrategy strat)
    {
        size_t elements = 1;

        for (auto d : dims)
            elements *= d;
        
        size_t bpe = 4;

        if (dtype == Core::DataType::FLOAT64 || dtype == Core::DataType::COMPLEX32) bpe = 8;
        else if (dtype == Core::DataType::COMPLEX64) bpe = 16;
        else if (dtype == Core::DataType::HALF) bpe = 2;

        auto storage = SushiRuntime::make_sushi<Storage>(context_.get_allocator(), elements * bpe, strat);
        
        Tensor t(storage, dims, 0, default_layout_);
        t.dtype = dtype;

        return t;
    }

} // namespace SushiBLAS
