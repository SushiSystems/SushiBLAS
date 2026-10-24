/**************************************************************************/
/* operation_options.cpp                                                  */
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
 * @file operation_options.cpp
 * @brief Defines the record-time check IndexOps, SignalOps and BlendOps apply to OperationOptions.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/operation_options.hpp>

#include <cstdint>

#include <SushiBLAS/engine/operand_check.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        void require_operation_options(const OperationOptions& options, const char* operation)
        {
            const std::uint8_t held = static_cast<std::uint8_t>(options.determinism);
            const std::uint8_t last =
                static_cast<std::uint8_t>(SushiRuntime::Determinism::Cosmetic);

            require_operand(!options.name.empty(), operation, "options", "names no task");
            require_operand(held <= last, operation, "options",
                            "holds a determinism class outside the enum");
        }
    } // namespace Dispatch
} // namespace SushiBLAS
