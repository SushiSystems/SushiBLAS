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
 * @brief Pins that the logger writes every line before an assertion aborts or a shutdown returns.
 * @author Mustafa Garip
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/core/logger.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Holds the line each test queues, short enough to count by its newline. */
            constexpr const char* K_LINE = "regression-logger-line\n";

            /** @brief Returns how many complete copies of K_LINE the captured text holds. */
            int count_lines(const std::string& captured)
            {
                const std::string line(K_LINE);
                int count = 0;

                for (size_t at = captured.find(line); at != std::string::npos;
                     at = captured.find(line, at + line.size()))
                    ++count;

                return count;
            }
        } // namespace

        TEST(Regression_LoggerDeathTest, AssertWritesItsMessageBeforeAborting)
        {
#ifdef NDEBUG
            GTEST_SKIP() << "SB_ASSERT compiles to nothing under NDEBUG; run this in a debug build.";
#else
            if (Core::K_ACTIVE_LOG_LEVEL < Core::LogLevel::Err)
                GTEST_SKIP() << "The error level is compiled out, so SB_ASSERT has no line to write.";

            constexpr int K_BACKLOG_LINES = 5000;

            EXPECT_DEATH(
                {
                    SB_LOGGER_SET_SYNC_MODE(false);

                    // The backlog keeps the worker busy, so only a flush can write the assertion.
                    for (int i = 0; i < K_BACKLOG_LINES; ++i)
                        Core::Detail::dispatch_log(Core::LogLevel::Err, std::string(K_LINE));

                    SB_ASSERT(false, "regression sentinel for the lost assertion message");
                },
                "Assertion failed: regression sentinel for the lost assertion message");
#endif
        }

        TEST(Regression_Logger, FlushReturnsAfterEveryQueuedLineIsWritten)
        {
            constexpr int K_LINES = 256;

            Core::Detail::AsyncLogger logger;

            ::testing::internal::CaptureStderr();
            for (int i = 0; i < K_LINES; ++i)
                logger.enqueue(Core::LogLevel::Err, std::string(K_LINE));
            logger.flush();
            const std::string captured = ::testing::internal::GetCapturedStderr();

            EXPECT_EQ(count_lines(captured), K_LINES)
                << "flush() returned while lines were still queued or being written.";
        }

        TEST(Regression_Logger, ShutdownWritesEveryQueuedLineAndReturns)
        {
            constexpr int K_ROUNDS = 2000;

            ::testing::internal::CaptureStderr();
            for (int round = 0; round < K_ROUNDS; ++round)
            {
                Core::Detail::AsyncLogger logger;
                logger.enqueue(Core::LogLevel::Err, std::string(K_LINE));
            }
            const std::string captured = ::testing::internal::GetCapturedStderr();

            EXPECT_EQ(count_lines(captured), K_ROUNDS)
                << "A logger destroyed right after a line was queued must still write that line.";
        }
    } // namespace Test
} // namespace SushiBLAS
