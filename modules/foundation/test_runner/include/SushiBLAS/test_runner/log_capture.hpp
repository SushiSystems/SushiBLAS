/**************************************************************************/
/* log_capture.hpp                                                        */
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
 * @file log_capture.hpp
 * @brief Declares LogCapture, which keeps the records SushiBLAS writes while a test runs.
 * @author Mustafa Garip
 * @see modules/foundation/test_runner/README.md
 */

#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <string_view>
#include <SushiLog/log/log_calls.hpp>
#include <SushiLog/log/log_level.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/sushiblas_log_project.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds the fields of one record a LogCapture received. */
        struct CapturedRecord
        {
            SushiLog::LogLevel level = SushiLog::LogLevel::Off; /**< The level the record was written at. */
            std::string module;                                  /**< The module of its category. */
            std::string message;                                 /**< The formatted message. */
        };

        /**
         * @brief Sends every record to a vector on the calling thread while it lives, at every level, and restores the
         *        default sink and level when it ends.
         *
         * A stand-in for the shipped capture sink SushiLog's backlog describes; a test that
         * asserts a record uses it until that sink exists.
         * @see modules/foundation/test_runner/README.md
         */
        class LogCapture final
        {
            public:
                /** @brief Switches SushiLog to synchronous delivery and starts keeping records. */
                LogCapture()
                {
                    SushiLog::set_log_synchronous(true);
                    SushiLog::set_log_level(K_SUSHIBLAS_LOG_PROJECT, SushiLog::LogLevel::Trace);
                    SushiLog::set_log_sink([this](const SushiLog::LogRecord& record)
                    {
                        records_.push_back({record.level, std::string(record.module),
                                            std::string(record.message)});
                    });
                }

                LogCapture(const LogCapture&) = delete;
                LogCapture& operator=(const LogCapture&) = delete;
                LogCapture(LogCapture&&) = delete;
                LogCapture& operator=(LogCapture&&) = delete;

                /** @brief Restores stderr as the sink and WARN as the level of SushiBLAS's records. */
                ~LogCapture()
                {
                    SushiLog::set_log_sink({});
                    SushiLog::set_log_level(K_SUSHIBLAS_LOG_PROJECT, SushiLog::LogLevel::Warn);
                }

                /** @brief Returns the records kept so far, in the order they were written. */
                [[nodiscard]] const std::vector<CapturedRecord>& records() const noexcept
                {
                    return records_;
                }

                /** @brief Returns how many kept records have @p level and a message containing @p text. */
                [[nodiscard]] std::size_t count(SushiLog::LogLevel level, std::string_view text) const
                {
                    std::size_t found = 0;
                    for (const CapturedRecord& record : records_)
                    {
                        if (record.level == level && record.message.find(text) != std::string::npos)
                            ++found;
                    }
                    return found;
                }

            private:
                /// @brief The records the sink received.
                std::vector<CapturedRecord> records_;
        };
    } // namespace Test
} // namespace SushiBLAS
