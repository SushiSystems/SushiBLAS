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
 * @brief Defines the Engine constructor, Engine::execute and Engine::discard_recorded.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine.hpp>

#include <cstddef>

#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/execution/execution_engine.hpp>

namespace SushiBLAS
{
    Engine::Engine(SushiRuntime::Execution::RuntimeContext& context, Core::Layout layout)
        : context_(context), scratch_pool_(context), graph_(context),
          recorder_(graph_, scratch_pool_), tensor_factory_(context, layout)
    {
        SB_LOG_INFO("SushiBLAS Engine initialized with {} layout.",
                    layout == Core::Layout::RowMajor ? "Row-Major" : "Column-Major");
    }

    void Engine::execute()
    {
        SushiRuntime::Execution::CompiledGraph compiled = graph_.compile();
        // An empty graph returns early; the execution engine throws on one; see include/SushiBLAS/README.md.
        if (compiled.roots.empty())
            return;

        context_.get_execution_engine().execute(compiled);
    }

    void Engine::discard_recorded()
    {
        // The nodes die with the returned vector, and take what their bodies captured with them.
        const std::size_t dropped = graph_.take_nodes().size();
        SB_LOG_INFO("SushiBLAS Engine discarded {} recorded task(s).", dropped);
    }
} // namespace SushiBLAS
