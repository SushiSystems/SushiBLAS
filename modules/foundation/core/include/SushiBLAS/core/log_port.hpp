/**************************************************************************/
/* log_port.hpp                                                           */
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
 * @file log_port.hpp
 * @brief Declares LogPort, the seam a unit logs through when a test must see its records, and SB_LOG_VIA.
 * @author Mustafa Garip
 * @see modules/foundation/core/README.md
 */

#pragma once

#include <string>

#include <SushiLog/log/log_category.hpp>
#include <SushiLog/log/log_level.hpp>
#include <SushiLog/text/text_format.hpp>
#include <SushiBLAS/core/logger.hpp>

namespace SushiBLAS
{
    namespace Core
    {
        /** @brief Receives the records of a unit that logs through SB_LOG_VIA instead of the process-wide logger. */
        class LogPort
        {
            public:
                LogPort() = default;
                LogPort(const LogPort&) = delete;
                LogPort& operator=(const LogPort&) = delete;
                LogPort(LogPort&&) = delete;
                LogPort& operator=(LogPort&&) = delete;
                virtual ~LogPort() = default;

                /** @brief Returns whether a record at @p level in @p category is kept. */
                [[nodiscard]] virtual bool admits(SushiLog::LogLevel level,
                                                  SushiLog::LogCategory category) const noexcept = 0;

                /** @brief Takes one formatted record written at @p file and @p line. */
                virtual void write(SushiLog::LogLevel level, SushiLog::LogCategory category,
                                   const char* file, int line, std::string&& message) = 0;
        };
    } // namespace Core
} // namespace SushiBLAS

/**
 * @brief Logs one record at @p level in @p category to @p port, a Core::LogPort, when the
 *        ceiling of this translation unit and the port both admit it.
 */
#define SB_LOG_VIA(port, level, category, ...)                                                       do                                                                                               {                                                                                                    if constexpr ((level) >= SushiBLAS::Core::K_LOG_LEVEL_CEILING)                                   {                                                                                                    if ((port).admits((level), (category)))                                                          (port).write((level), (category), __FILE__, __LINE__,                                                        SushiLog::format_text(__VA_ARGS__));                                      }                                                                                            } while (0)
