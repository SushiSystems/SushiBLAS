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
     *
     * IndexOps, SignalOps and BlendOps take it; the other families take a name and an
     * OperationID.
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
