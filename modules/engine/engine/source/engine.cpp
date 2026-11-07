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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file engine.cpp
 * @brief Defines the Engine constructors.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/engine.hpp>

#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/engine_log_category.hpp>
#include <SushiBLAS/engine/host_scratch_source.hpp>
#include <SushiBLAS/engine/host_task_sink.hpp>
#include <SushiBLAS/engine/profile_hooks.hpp>

namespace SushiBLAS
{
    Engine::Engine(SushiRuntime::Runtime& runtime, Core::Layout layout)
        : runtime_(&runtime), scratch_pool_(runtime), tensor_factory_(runtime, layout)
    {
        Internal::install_profile_hooks();
        SB_LOG_INFO(K_ENGINE_LOG_CATEGORY, "SushiBLAS Engine initialized with {} layout.",
                    layout == Core::Layout::RowMajor ? "Row-Major" : "Column-Major");
    }

    Engine::~Engine() = default;

    Engine::Engine(Host, Core::Layout layout)
        : runtime_(nullptr), host_source_(std::make_unique<HostScratchSource>()),
          scratch_pool_(*host_source_), mode_(ExecutionMode::ForceReference),
          tensor_factory_(TensorFactory::Host{}, layout),
          host_sink_(std::make_unique<HostTaskSink>())
    {
        SB_LOG_INFO(K_ENGINE_LOG_CATEGORY, "SushiBLAS host Engine initialized with {} layout.",
                    layout == Core::Layout::RowMajor ? "Row-Major" : "Column-Major");
    }
} // namespace SushiBLAS
