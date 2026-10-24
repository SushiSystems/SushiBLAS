/**************************************************************************/
/* index_options.hpp                                                      */
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

/**
 * @file index_options.hpp
 * @brief Declares IndexOptions, what a caller states about one per-index task.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <SushiRuntime/core/determinism.hpp>

namespace SushiBLAS
{
    /**
     * @brief Holds the name of a per-index task and the determinism contract its caller states.
     * @see docs/design/LAYERED_EXECUTION/README.md
     */
    struct IndexOptions
    {
        /** @brief The label the task reports under; it must not be empty. */
        std::string_view name;

        /** @brief The contract the task's node carries to the runtime's plan. */
        SushiRuntime::Determinism determinism = SushiRuntime::Determinism::Unstated;
    };
} // namespace SushiBLAS
