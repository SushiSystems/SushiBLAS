/**************************************************************************/
/* test_captured_record_count.cpp                                         */
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
 * @file test_captured_record_count.cpp
 * @brief Tests that count_records counts the records of one level whose message holds a text.
 * @author Mustafa Garip
 * @see tests/foundation/test_runner/README.md
 */

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <SushiBLAS/test_runner/captured_record_count.hpp>
#include <SushiLog/log/log_capture.hpp>
#include <SushiLog/log/log_level.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Builds a captured record of @p level with @p message and no other field set. */
            [[nodiscard]] SushiLog::CapturedRecord record_of(SushiLog::LogLevel level, std::string message)
            {
                SushiLog::CapturedRecord record{};
                record.level = level;
                record.message = std::move(message);
                return record;
            }
        } // namespace

        TEST(Small_CapturedRecordCount, CountsOnlyTheRecordsOfTheLevelThatHoldTheText)
        {
            const std::vector<SushiLog::CapturedRecord> records = {
                record_of(SushiLog::LogLevel::Warn, "operand a is empty"),
                record_of(SushiLog::LogLevel::Warn, "operand b is empty"),
                record_of(SushiLog::LogLevel::Error, "operand c is empty"),
                record_of(SushiLog::LogLevel::Warn, "queue drained"),
            };

            EXPECT_EQ(count_records(records, SushiLog::LogLevel::Warn, "is empty"), std::size_t{2});
        }

        TEST(Small_CapturedRecordCount, CountsNothingInNoRecords)
        {
            const std::vector<SushiLog::CapturedRecord> records;

            EXPECT_EQ(count_records(records, SushiLog::LogLevel::Warn, ""), std::size_t{0});
        }
    } // namespace Test
} // namespace SushiBLAS
