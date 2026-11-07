/**************************************************************************/
/* mode_admission.hpp                                                     */
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
 * @file mode_admission.hpp
 * @brief Declares ModeAdmission, the port an Engine asks which execution modes it may start a
 *        recorder in.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/engine/execution_mode.hpp>

namespace SushiBLAS
{
    /**
     * @brief Answers which execution modes an Engine may start a recorder in.
     *
     * HostModeAdmission admits ForceReference only; RuntimeModeAdmission admits every mode. An
     * Engine owns one, given by its constructor, and Engine::set_mode asks it, the same call for
     * both kinds of engine.
     * @see modules/engine/engine/README.md
     */
    class ModeAdmission
    {
        public:
            ModeAdmission(const ModeAdmission&) = delete;
            ModeAdmission& operator=(const ModeAdmission&) = delete;
            ModeAdmission(ModeAdmission&&) = delete;
            ModeAdmission& operator=(ModeAdmission&&) = delete;
            virtual ~ModeAdmission() = default;

            /** @brief Reports whether a recorder may start in @p mode. */
            [[nodiscard]] virtual bool admits(ExecutionMode mode) const noexcept = 0;

        protected:
            /** @brief Lets an adapter construct itself; the port is never built alone. */
            ModeAdmission() = default;
    };
} // namespace SushiBLAS
