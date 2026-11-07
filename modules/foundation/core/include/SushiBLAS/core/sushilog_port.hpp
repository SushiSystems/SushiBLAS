/**************************************************************************/
/* sushilog_port.hpp                                                      */
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
 * @file sushilog_port.hpp
 * @brief Declares SushiLogPort, the LogPort that hands its records to the process-wide SushiLog.
 * @author Mustafa Garip
 * @see modules/foundation/core/README.md
 */

#pragma once

#include <string>
#include <utility>

#include <SushiLog/log/log_calls.hpp>
#include <SushiLog/log/log_front.hpp>
#include <SushiBLAS/core/log_port.hpp>

namespace SushiBLAS
{
    namespace Core
    {
        /** @brief Sends each record through the level and the sink SushiLog holds for the process. */
        class SushiLogPort final : public LogPort
        {
            public:
                /** @brief Returns whether SushiLog's level for @p category lets @p level through. */
                [[nodiscard]] bool admits(SushiLog::LogLevel level,
                                          SushiLog::LogCategory category) const noexcept override
                {
                    return level >= SushiLog::log_level(category);
                }

                /** @brief Hands the record to SushiLog's logger. */
                void write(SushiLog::LogLevel level, SushiLog::LogCategory category,
                           const char* file, int line, std::string&& message) override
                {
                    SushiLog::Detail::emit_log_record(level, category, file, line, std::move(message));
                }
        };
    } // namespace Core
} // namespace SushiBLAS
