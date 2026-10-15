/**************************************************************************/
/* execution_mode_sweep.hpp                                               */
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
 * @file execution_mode_sweep.hpp
 * @brief Declares K_BOTH_MODES, the two execution modes a routine's tests run it in.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/engine/execution_mode.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The two execution modes every sparse routine is tested in.
        constexpr ExecutionMode K_BOTH_MODES[2] = {ExecutionMode::ForceReference,
                                                   ExecutionMode::ForceSYCL};
    } // namespace Test
} // namespace SushiBLAS
