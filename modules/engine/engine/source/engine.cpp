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
 * @brief Defines the Engine constructor.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/engine.hpp>

#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/engine_log_category.hpp>
#include <SushiBLAS/engine/profile_hooks.hpp>

namespace SushiBLAS
{
    Engine::Engine(SushiRuntime::Runtime& runtime, Core::Layout layout)
        : runtime_(runtime), scratch_pool_(runtime), tensor_factory_(runtime, layout)
    {
        Internal::install_profile_hooks();
        SB_LOG_INFO(K_ENGINE_LOG_CATEGORY, "SushiBLAS Engine initialized with {} layout.",
                    layout == Core::Layout::RowMajor ? "Row-Major" : "Column-Major");
    }
} // namespace SushiBLAS
