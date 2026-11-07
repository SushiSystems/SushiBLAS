/**************************************************************************/
/* host_mode_admission.hpp                                                */
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
 * @file host_mode_admission.hpp
 * @brief Declares HostModeAdmission, the ModeAdmission of a host engine: ForceReference only.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/mode_admission.hpp>

namespace SushiBLAS
{
    /**
     * @brief Admits ForceReference only: a host engine has no SYCL queue to run another mode on.
     * @see modules/engine/engine/README.md
     */
    class HostModeAdmission final : public ModeAdmission
    {
        public:
            HostModeAdmission() = default;

            /** @brief Reports true for ForceReference only. */
            [[nodiscard]] bool admits(ExecutionMode mode) const noexcept override
            {
                return mode == ExecutionMode::ForceReference;
            }
    };
} // namespace SushiBLAS
