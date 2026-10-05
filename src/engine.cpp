/****************************************************************/
/* engine.cpp                                                   */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
