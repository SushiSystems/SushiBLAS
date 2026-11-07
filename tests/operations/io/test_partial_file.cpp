/**************************************************************************/
/* test_partial_file.cpp                                                  */
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
 * @file test_partial_file.cpp
 * @brief Tests that remove_partial_file reports a removed file quietly and warns when it cannot remove one.
 * @author Mustafa Garip
 * @see tests/operations/io/README.md
 */

#include <cstddef>
#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/io/partial_file.hpp>
#include <SushiLog/log/log_capture.hpp>
#include <SushiBLAS/test_runner/captured_record_count.hpp>
#include <SushiBLAS/test_runner/scoped_log_level.hpp>
#include "support/memory_file_remover.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The path the tests hold in memory; no file of this name is made.
            const std::string K_PARTIAL_PATH = "partial_save.bin";
        } // namespace

        TEST(Small_IoLog, ARemovedPartialFileIsReportedGone)
        {
            MemoryFileRemover remover;
            remover.add(K_PARTIAL_PATH);

            EXPECT_TRUE(IOFiles::remove_partial_file(K_PARTIAL_PATH, remover));
            EXPECT_FALSE(remover.has(K_PARTIAL_PATH));
        }

        TEST(Small_IoLog, ARemovedPartialFileStatesNothing)
        {
            MemoryFileRemover remover;
            remover.add(K_PARTIAL_PATH);
            const ScopedLogLevel level(SushiLog::LogLevel::Trace);
            const SushiLog::LogCapture capture;

            static_cast<void>(IOFiles::remove_partial_file(K_PARTIAL_PATH, remover));

            EXPECT_TRUE(capture.records().empty());
        }

        TEST(Small_IoLog, AFailedCleanupWarns)
        {
            const MemoryFileRemover remover;
            const SushiLog::LogCapture capture;

            EXPECT_FALSE(IOFiles::remove_partial_file(K_PARTIAL_PATH, remover));

            EXPECT_EQ(count_records(capture.records(), SushiLog::LogLevel::Warn,
                                    "could not remove the partial file"),
                      std::size_t{1});
            EXPECT_EQ(count_records(capture.records(), SushiLog::LogLevel::Warn, K_PARTIAL_PATH),
                      std::size_t{1});
        }
    } // namespace Test
} // namespace SushiBLAS
