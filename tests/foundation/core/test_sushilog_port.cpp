/**************************************************************************/
/* test_sushilog_port.cpp                                                 */
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
 * @file test_sushilog_port.cpp
 * @brief Tests SushiLogPort and the SB_LOG macros against the process-wide SushiLog logger.
 * @author Mustafa Garip
 * @see tests/foundation/core/README.md
 */

#include <cstddef>
#include <stdexcept>

#include <gtest/gtest.h>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/sushilog_port.hpp>
#include <SushiBLAS/core/sushiblas_log_project.hpp>
#include <SushiLog/log/log_capture.hpp>
#include <SushiLog/log/log_category.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Names the records these tests write, as a module of SushiBLAS would. */
            constexpr SushiLog::LogCategory K_PROCESS_PROBE_LOG_CATEGORY =
                SushiLog::make_log_category(K_SUSHIBLAS_LOG_PROJECT, "probe");
        } // namespace

        TEST(Medium_SushiLogPort, ARecordOfTheMacroReachesTheSinkWithItsProjectAndModule)
        {
            const SushiLog::LogCapture capture;

            SB_LOG_ERROR(K_PROCESS_PROBE_LOG_CATEGORY, "probe {} of {}", 1, 2);

            ASSERT_EQ(capture.records().size(), std::size_t{1});
            EXPECT_EQ(capture.records()[0].level, SushiLog::LogLevel::Error);
            EXPECT_EQ(capture.records()[0].project, "sushiblas");
            EXPECT_EQ(capture.records()[0].module, "probe");
            EXPECT_EQ(capture.records()[0].message, "probe 1 of 2");
        }

        TEST(Medium_SushiLogPort, ARecordOfTheSushiLogPortReachesTheSink)
        {
            const SushiLog::LogCapture capture;
            Core::SushiLogPort port;

            SB_LOG_VIA(port, SushiLog::LogLevel::Error, K_PROCESS_PROBE_LOG_CATEGORY, "via {}", "port");

            ASSERT_EQ(capture.records().size(), std::size_t{1});
            EXPECT_EQ(capture.records()[0].module, "probe");
            EXPECT_EQ(capture.records()[0].message, "via port");
        }

        TEST(Medium_SushiLogPort, ThrowIfLogsNothing)
        {
            const SushiLog::LogCapture capture;

            EXPECT_THROW(SB_THROW_IF(true, "operand {} is empty", 3), std::runtime_error);

            EXPECT_TRUE(capture.records().empty());
        }

#ifndef NDEBUG
        // SB_ASSERT compiles to nothing under NDEBUG, so the death test exists in debug builds only.
        TEST(Medium_SushiLogPortDeathTest, AssertWritesAFatalRecordInItsCategoryBeforeAborting)
        {
            EXPECT_DEATH(SB_ASSERT(K_PROCESS_PROBE_LOG_CATEGORY, false,
                                   "regression sentinel for the lost assertion message"),
                         "sushiblas.probe.*regression sentinel for the lost assertion message");
        }
#endif
    } // namespace Test
} // namespace SushiBLAS
