/**************************************************************************/
/* operation_options.hpp                                                  */
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
 * @file operation_options.hpp
 * @brief Declares OperationOptions, what a caller states about the node one operation records.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <SushiRuntime/core/determinism.hpp>

namespace SushiBLAS
{
    /**
     * @brief Holds the name of a recorded node and the determinism contract its caller states.
     * @see docs/design/LAYERED_EXECUTION/README.md
     */
    struct OperationOptions
    {
        /** @brief The label the node reports under; it must not be empty. */
        std::string_view name;

        /** @brief The contract the node carries to the runtime's plan. */
        SushiRuntime::Determinism determinism = SushiRuntime::Determinism::Unstated;
    };

    namespace Dispatch
    {
        /**
         * @brief Throws unless options names the node and holds a determinism class of the runtime.
         * @param operation The operation id the message names.
         * @throws std::invalid_argument for an empty name, or a class outside the enum.
         */
        void require_operation_options(const OperationOptions& options, const char* operation);
    } // namespace Dispatch
} // namespace SushiBLAS
