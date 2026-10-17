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

        TEST(Medium_Logger, ThrowIfThrowsTheFormattedMessageAndLogsNothing)
        {
            CapturingSink sink;

            try
            {
                SB_THROW_IF(true, "operand {} is empty", 3);
                FAIL() << "SB_THROW_IF did not throw.";
            }
            catch (const std::runtime_error& error)
            {
                EXPECT_STREQ(error.what(), "operand 3 is empty");
            }

            EXPECT_TRUE(sink.records.empty());
        }

        TEST(Medium_LoggerDeathTest, AssertWritesAFatalRecordInItsCategoryBeforeAborting)
        {
#ifdef NDEBUG
            GTEST_SKIP() << "SB_ASSERT compiles to nothing under NDEBUG; run this in a debug build.";
#else
            EXPECT_DEATH(SB_ASSERT(K_PROBE_LOG_CATEGORY, false,
                                   "regression sentinel for the lost assertion message"),
                         "sushiblas.probe.*regression sentinel for the lost assertion message");
#endif
        }
    } // namespace Test
} // namespace SushiBLAS
