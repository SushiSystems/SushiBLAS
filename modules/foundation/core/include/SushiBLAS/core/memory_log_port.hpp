/**************************************************************************/
/* memory_log_port.hpp                                                    */
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
 * @file memory_log_port.hpp
 * @brief Declares MemoryLogPort, the LogPort that keeps its records in memory.
 * @author Mustafa Garip
 * @see modules/foundation/core/README.md
 */

#pragma once

#include <string>
#include <vector>

#include <SushiBLAS/core/log_port.hpp>

namespace SushiBLAS
{
    namespace Core
    {
        /** @brief Holds the fields of one record a MemoryLogPort kept. */
        struct MemoryLogRecord
        {
            SushiLog::LogLevel level = SushiLog::LogLevel::Off; /**< The level the record was written at. */
            std::string project;                                 /**< The project of its category. */
            std::string module;                                  /**< The module of its category. */
            std::string message;                                 /**< The formatted message. */
        };

        /**
         * @brief Keeps every record at or above a level, in the order written, and touches no
         *        process-wide state.
         */
        class MemoryLogPort final : public LogPort
        {
            public:
                /** @brief Keeps the records at or above @p level; every level by default. */
                explicit MemoryLogPort(SushiLog::LogLevel level = SushiLog::LogLevel::Trace) noexcept
                    : level_(level)
                {

                }

                /** @brief Changes the lowest level this port keeps to @p level. */
                void set_level(SushiLog::LogLevel level) noexcept
                {
                    level_ = level;
                }

                /** @brief Returns whether @p level is at or above the level of this port. */
                [[nodiscard]] bool admits(SushiLog::LogLevel level,
                                          SushiLog::LogCategory category) const noexcept override
                {
                    static_cast<void>(category);
                    return level >= level_;
                }

                /** @brief Keeps the record. */
                void write(SushiLog::LogLevel level, SushiLog::LogCategory category,
                           const char* file, int line, std::string&& message) override
                {
                    static_cast<void>(file);
                    static_cast<void>(line);
                    records_.push_back({level, std::string(category.project().name()),
                                        std::string(category.module()), std::move(message)});
                }

                /** @brief Returns the records kept so far, in the order written. */
                [[nodiscard]] const std::vector<MemoryLogRecord>& records() const noexcept
                {
                    return records_;
                }

            private:
                /// @brief The lowest level kept.
                SushiLog::LogLevel level_;
                /// @brief The records kept.
                std::vector<MemoryLogRecord> records_;
        };
    } // namespace Core
} // namespace SushiBLAS
