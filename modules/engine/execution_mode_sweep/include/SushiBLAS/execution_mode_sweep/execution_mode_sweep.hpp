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
