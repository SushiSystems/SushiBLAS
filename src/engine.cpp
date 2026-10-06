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
 * @file engine.cpp
 * @brief Defines the Engine constructor and its tensor creation.
 * @author Mustafa Garip
 */

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
                                 const SushiRuntime::Memory::IAllocatorStrategy& strat)
    {
        return create_tensor(dims, Core::DataType::FLOAT32, strat);
    }

    Tensor Engine::create_tensor(std::initializer_list<int64_t> dims,
                                 Core::DataType dtype,
                                 const SushiRuntime::Memory::IAllocatorStrategy& strat)
    {
        size_t elements = 1;

        for (auto d : dims)
            elements *= d;
        
        const size_t bpe = Core::bytes_per_element(dtype);

        auto storage = SushiRuntime::make_sushi<Storage>(context_.get_allocator(), elements * bpe, strat);
        
        return Tensor(storage, dims, dtype, 0, default_layout_);
    }

} // namespace SushiBLAS
