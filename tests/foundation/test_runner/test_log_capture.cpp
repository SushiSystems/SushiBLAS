/**************************************************************************/
/* test_log_capture.cpp                                                   */
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
 * @file test_log_capture.cpp
 * @brief Tests that LogCapture keeps the records written while it lives and restores the logger after.
 * @author Mustafa Garip
 * @see tests/foundation/test_runner/README.md
 */

#include <gtest/gtest.h>
#include <SushiLog/log/log_category.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/test_runner/log_capture.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Names the records these tests write. */
            constexpr SushiLog::LogCategory K_CAPTURE_LOG_CATEGORY =
                SushiLog::make_log_category(K_SUSHIBLAS_LOG_PROJECT, "capture");
        } // namespace

        TEST(Medium_LogCapture, KeepsEveryRecordWithItsLevelModuleAndMessage)
        {
            LogCapture capture;

            SB_LOG_INFO(K_CAPTURE_LOG_CATEGORY, "first {}", 1);
            SB_LOG_WARN(K_CAPTURE_LOG_CATEGORY, "second {}", 2);

            ASSERT_EQ(capture.records().size(), std::size_t{2});
            EXPECT_EQ(capture.records()[0].level, SushiLog::LogLevel::Info);
            EXPECT_EQ(capture.records()[0].module, "capture");
            EXPECT_EQ(capture.records()[0].message, "first 1");
            EXPECT_EQ(capture.records()[1].message, "second 2");
        }

        TEST(Medium_LogCapture, CountsTheRecordsOfOneLevelThatContainAText)
        {
            LogCapture capture;

            SB_LOG_WARN(K_CAPTURE_LOG_CATEGORY, "device A fell back");
            SB_LOG_WARN(K_CAPTURE_LOG_CATEGORY, "device B fell back");
            SB_LOG_INFO(K_CAPTURE_LOG_CATEGORY, "device A fell back");

            EXPECT_EQ(capture.count(SushiLog::LogLevel::Warn, "device A"), std::size_t{1});
            EXPECT_EQ(capture.count(SushiLog::LogLevel::Warn, "fell back"), std::size_t{2});
            EXPECT_EQ(capture.count(SushiLog::LogLevel::Error, "fell back"), std::size_t{0});
        }

        TEST(Medium_LogCapture, ARecordAfterItEndsIsNotKept)
        {
            {
                LogCapture capture;
                SB_LOG_WARN(K_CAPTURE_LOG_CATEGORY, "inside");
                EXPECT_EQ(capture.records().size(), std::size_t{1});
            }

            LogCapture later;
            EXPECT_TRUE(later.records().empty());
        }
    } // namespace Test
} // namespace SushiBLAS
