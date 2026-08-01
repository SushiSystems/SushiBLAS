/**************************************************************************/
/* engine.cpp                                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
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
        
        return Tensor(storage, dims, dtype, 0, default_layout_);
    }

} // namespace SushiBLAS
