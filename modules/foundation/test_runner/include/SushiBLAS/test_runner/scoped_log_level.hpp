/**************************************************************************/
/* scoped_log_level.hpp                                                   */
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
 * @file scoped_log_level.hpp
 * @brief Declares ScopedLogLevel, which sets the level of SushiBLAS's records for one scope.
 * @author Mustafa Garip
 * @see modules/foundation/test_runner/README.md
 */

#pragma once

#include <SushiLog/log/log_calls.hpp>
#include <SushiLog/log/log_level.hpp>
#include <SushiBLAS/core/sushiblas_log_project.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Sets the level of every SushiBLAS category while it lives and puts back the level
         *        it found when it ends.
         *
         * A test that asserts a record below the default level holds one beside a SushiLog::LogCapture.
         */
        class ScopedLogLevel final
        {
            public:
                /** @brief Sets @p level as the level of SushiBLAS's project. */
                explicit ScopedLogLevel(SushiLog::LogLevel level) noexcept
                    : previous_(SushiLog::log_level(K_SUSHIBLAS_LOG_PROJECT))
                {
                    SushiLog::set_log_level(K_SUSHIBLAS_LOG_PROJECT, level);
                }

                ScopedLogLevel(const ScopedLogLevel&) = delete;
                ScopedLogLevel& operator=(const ScopedLogLevel&) = delete;
                ScopedLogLevel(ScopedLogLevel&&) = delete;
                ScopedLogLevel& operator=(ScopedLogLevel&&) = delete;

                /** @brief Puts back the level found on entry. */
                ~ScopedLogLevel()
                {
                    SushiLog::set_log_level(K_SUSHIBLAS_LOG_PROJECT, previous_);
                }

            private:
                /// @brief The project level found on entry.
                SushiLog::LogLevel previous_;
        };
    } // namespace Test
} // namespace SushiBLAS
