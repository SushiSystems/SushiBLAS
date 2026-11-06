/**************************************************************************/
/* test_logger.cpp                                                        */
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
 * @file test_logger.cpp
 * @brief Tests the SB_LOG macros, SB_ASSERT and SB_THROW_IF over SushiLog.
 * @author Mustafa Garip
 */

#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <SushiBLAS/core/logger.hpp>
#include <SushiLog/log/log_calls.hpp>
#include <SushiLog/log/log_category.hpp>
#include <SushiBLAS/core/sushiblas_log_project.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Names the records these tests write, as a module of SushiBLAS would. */
            constexpr SushiLog::LogCategory K_PROBE_LOG_CATEGORY =
                SushiLog::make_log_category(K_SUSHIBLAS_LOG_PROJECT, "probe");

            /** @brief Holds the fields of one record a test's sink received. */
            struct CapturedRecord
            {
                SushiLog::LogLevel level;
                std::string project;
                std::string module;
                std::string message;
            };

            /**
             * @brief Sends every record to a vector while it lives, on the calling thread, and
             *        restores stderr and the project's level when it ends.
             */
            class CapturingSink
            {
                public:
                    CapturingSink()
                    {
                        SushiLog::set_log_synchronous(true);
                        SushiLog::set_log_level(K_SUSHIBLAS_LOG_PROJECT, SushiLog::LogLevel::Trace);
                        SushiLog::set_log_sink([this](const SushiLog::LogRecord& record)
                        {
                            records.push_back({record.level, std::string(record.project),
                                               std::string(record.module), std::string(record.message)});
                        });
                    }

                    ~CapturingSink()
                    {
                        SushiLog::set_log_sink({});
                        SushiLog::set_log_level(K_SUSHIBLAS_LOG_PROJECT, SushiLog::LogLevel::Warn);
                    }

                    CapturingSink(const CapturingSink&) = delete;
                    CapturingSink& operator=(const CapturingSink&) = delete;

                    std::vector<CapturedRecord> records;
            };

            /** @brief Returns the what() of the runtime_error thrower throws, or "" if none. */
            template<typename Thrower>
            [[nodiscard]] std::string message_thrown_by(Thrower thrower)
            {
                try
                {
                    thrower();
                }
                catch (const std::runtime_error& error)
                {
                    return error.what();
                }
                return {};
            }
        } // namespace

        TEST(Medium_Logger, ARecordCarriesTheSushiBLASProjectAndItsModule)
        {
            CapturingSink sink;

            SB_LOG_WARN(K_PROBE_LOG_CATEGORY, "probe {} of {}", 1, 2);

            ASSERT_EQ(sink.records.size(), 1u);
            EXPECT_EQ(sink.records[0].level, SushiLog::LogLevel::Warn);
            EXPECT_EQ(sink.records[0].project, "sushiblas");
            EXPECT_EQ(sink.records[0].module, "probe");
            EXPECT_EQ(sink.records[0].message, "probe 1 of 2");
        }

        TEST(Medium_Logger, ARecordBelowTheProjectLevelIsDropped)
        {
            CapturingSink sink;
            SushiLog::set_log_level(K_SUSHIBLAS_LOG_PROJECT, SushiLog::LogLevel::Error);

            SB_LOG_WARN(K_PROBE_LOG_CATEGORY, "below the level");

            EXPECT_TRUE(sink.records.empty());
        }

        TEST(Medium_Logger, ThrowIfThrowsTheFormattedMessage)
        {
            const auto thrower = [] { SB_THROW_IF(true, "operand {} is empty", 3); };

            EXPECT_EQ(message_thrown_by(thrower), "operand 3 is empty");
        }

        TEST(Medium_Logger, ThrowIfThrowsARuntimeError)
        {
            EXPECT_THROW(SB_THROW_IF(true, "operand {} is empty", 3), std::runtime_error);
        }

        TEST(Medium_Logger, ThrowIfLogsNothing)
        {
            CapturingSink sink;

            EXPECT_THROW(SB_THROW_IF(true, "operand {} is empty", 3), std::runtime_error);

            EXPECT_TRUE(sink.records.empty());
        }

#ifndef NDEBUG
        // SB_ASSERT compiles to nothing under NDEBUG, so the death test exists in debug builds only.
        TEST(Medium_LoggerDeathTest, AssertWritesAFatalRecordInItsCategoryBeforeAborting)
        {
            EXPECT_DEATH(SB_ASSERT(K_PROBE_LOG_CATEGORY, false,
                                   "regression sentinel for the lost assertion message"),
                         "sushiblas.probe.*regression sentinel for the lost assertion message");
        }
#endif
    } // namespace Test
} // namespace SushiBLAS
