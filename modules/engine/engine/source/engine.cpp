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

#include <SushiBLAS/engine/engine.hpp>

#include <cstddef>
#include <utility>

#include <SushiBLAS/core/logger.hpp>

namespace SushiBLAS
{
    Engine::Engine(SushiRuntime::Runtime& runtime, Core::Layout layout)
        : runtime_(runtime), scratch_pool_(runtime), graph_(runtime.graph()),
          recorder_(graph_.recorder(), scratch_pool_), tensor_factory_(runtime, layout)
    {
        SB_LOG_INFO("SushiBLAS Engine initialized with {} layout.",
                    layout == Core::Layout::RowMajor ? "Row-Major" : "Column-Major");
    }

    SushiRuntime::Result<void> Engine::execute()
    {
        SushiRuntime::Recorder& recorded = graph_.recorder();
        // A graph with no task is not run: the runtime refuses one; see modules/README.md.
        if (recorded.node_count() == 0 && recorded.refusal() == nullptr)
            return {};

        auto report = graph_.run();
        // A new graph replaces the one that ran, so its tasks die with what they captured.
        graph_ = runtime_.graph();
        if (!report)
            return SushiRuntime::make_error(std::move(report.error()));

        return {};
    }

    void Engine::discard_recorded()
    {
        const std::size_t dropped = graph_.recorder().node_count();
        // The tasks die in clear(), and take what their bodies captured with them.
        graph_.recorder().clear();
        SB_LOG_INFO("SushiBLAS Engine discarded {} recorded task(s).", dropped);
    }
} // namespace SushiBLAS
