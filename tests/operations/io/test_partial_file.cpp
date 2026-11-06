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
 * @brief Tests that remove_partial_file deletes a file, and warns when it cannot.
 * @author Mustafa Garip
 * @see tests/operations/io/README.md
 */

#include <string>
#include <fstream>
#include <filesystem>
#include <gtest/gtest.h>
#include <SushiBLAS/io/partial_file.hpp>
#include <SushiBLAS/test_runner/log_capture.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Medium_IoLog, ARemovedPartialFileIsGoneAndStatesNothing)
        {
            const std::filesystem::path path =
                std::filesystem::temp_directory_path() / "sushiblas_io_partial_file.bin";
            std::ofstream(path, std::ios::binary) << "partial";
            LogCapture capture;

            EXPECT_TRUE(IOFiles::remove_partial_file(path.string()));

            EXPECT_FALSE(std::filesystem::exists(path));
            EXPECT_TRUE(capture.records().empty());
        }

        TEST(Medium_IoLog, AFailedCleanupWarns)
        {
            const std::filesystem::path path =
                std::filesystem::temp_directory_path() / "sushiblas_io_no_such_partial_file.bin";
            std::filesystem::remove(path);
            LogCapture capture;

            EXPECT_FALSE(IOFiles::remove_partial_file(path.string()));

            EXPECT_EQ(capture.count(SushiLog::LogLevel::Warn, "could not remove the partial file"),
                      std::size_t{1});
            EXPECT_EQ(capture.count(SushiLog::LogLevel::Warn, path.string()), std::size_t{1});
        }
    } // namespace Test
} // namespace SushiBLAS
