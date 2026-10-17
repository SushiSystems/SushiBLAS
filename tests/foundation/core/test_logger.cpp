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
 * @brief Tests that the SB_LOG macros, SB_ASSERT and SB_THROW_IF write through SushiLog.
 * @author Mustafa Garip
 */

#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <SushiBLAS/core/logger.hpp>
#include <SushiLog/log/log_calls.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
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

            SB_LOG_WARN(K_CORE_LOG_CATEGORY, "probe {} of {}", 1, 2);

            ASSERT_EQ(sink.records.size(), 1u);
            EXPECT_EQ(sink.records[0].level, SushiLog::LogLevel::Warn);
            EXPECT_EQ(sink.records[0].project, "sushiblas");
            EXPECT_EQ(sink.records[0].module, "core");
            EXPECT_EQ(sink.records[0].message, "probe 1 of 2");
        }

        TEST(Medium_Logger, ARecordBelowTheProjectLevelIsDropped)
        {
            CapturingSink sink;
            SushiLog::set_log_level(K_SUSHIBLAS_LOG_PROJECT, SushiLog::LogLevel::Error);

            SB_LOG_WARN(K_CORE_LOG_CATEGORY, "below the level");

            EXPECT_TRUE(sink.records.empty());
        }

        TEST(Medium_Logger, ThrowIfLogsAnErrorAndThrowsTheSameMessage)
        {
            CapturingSink sink;

            EXPECT_THROW(SB_THROW_IF(true, "operand {} is empty", 3), std::runtime_error);

            ASSERT_EQ(sink.records.size(), 1u);
            EXPECT_EQ(sink.records[0].level, SushiLog::LogLevel::Error);
            EXPECT_EQ(sink.records[0].message, "throwing std::runtime_error: operand 3 is empty");
        }

        TEST(Medium_LoggerDeathTest, AssertWritesAFatalRecordBeforeAborting)
        {
#ifdef NDEBUG
            GTEST_SKIP() << "SB_ASSERT compiles to nothing under NDEBUG; run this in a debug build.";
#else
            EXPECT_DEATH(SB_ASSERT(false, "regression sentinel for the lost assertion message"),
                         "regression sentinel for the lost assertion message");
#endif
        }
    } // namespace Test
} // namespace SushiBLAS
