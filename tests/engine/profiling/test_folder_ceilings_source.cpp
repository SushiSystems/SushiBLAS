/**************************************************************************/
/* test_folder_ceilings_source.cpp                                        */
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
 * @file test_folder_ceilings_source.cpp
 * @brief Tests that FolderCeilingsSource reads the bytes of a file in its folder and reads nothing for a file that is not there.
 * @author Mustafa Garip
 * @see tests/engine/profiling/README.md
 */

#include <string>
#include <fstream>
#include <filesystem>
#include <gtest/gtest.h>
#include <SushiBLAS/profiling/folder_ceilings_source.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Owns a folder under the temporary directory and removes it afterwards. */
            class Medium_FolderCeilingsSource : public ::testing::Test
            {
                protected:
                    void SetUp() override
                    {
                        const std::string name = std::string("sushiblas_folder_source_") +
                            ::testing::UnitTest::GetInstance()->current_test_info()->name();
                        folder_ = std::filesystem::temp_directory_path() / name;
                        std::filesystem::remove_all(folder_);
                        std::filesystem::create_directories(folder_);
                    }

                    void TearDown() override
                    {
                        std::filesystem::remove_all(folder_);
                    }

                    /// @brief The folder the files of one test live in.
                    std::filesystem::path folder_;
            };
        } // namespace

        TEST_F(Medium_FolderCeilingsSource, ReadsTheBytesOfAFileInTheFolder)
        {
            std::ofstream file(folder_ / "a.ceilings.json", std::ios::binary);
            file << "line one\nline two";
            file.close();
            const Profiling::FolderCeilingsSource source(folder_.string());

            EXPECT_EQ(source.read("a.ceilings.json"), "line one\nline two");
        }

        TEST_F(Medium_FolderCeilingsSource, ReadsNothingForAFileThatIsNotInTheFolder)
        {
            const Profiling::FolderCeilingsSource source(folder_.string());

            EXPECT_FALSE(source.read("a.ceilings.json").has_value());
        }

        TEST_F(Medium_FolderCeilingsSource, LocatesAFileInsideTheFolder)
        {
            const Profiling::FolderCeilingsSource source(folder_.string());

            EXPECT_EQ(std::filesystem::path(source.location("a.ceilings.json")), folder_ / "a.ceilings.json");
        }
    } // namespace Test
} // namespace SushiBLAS
