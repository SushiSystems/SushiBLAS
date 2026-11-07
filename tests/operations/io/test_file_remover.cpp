/**************************************************************************/
/* test_file_remover.cpp                                                  */
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
 * @file test_file_remover.cpp
 * @brief Tests that the disk and in-memory FileRemover adapters delete a file they hold and refuse one they do not.
 * @author Mustafa Garip
 * @see tests/operations/io/README.md
 */

#include <filesystem>
#include <fstream>
#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/io/file_remover.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include "support/memory_file_remover.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief The in-memory adapter and the file it holds. */
            struct MemorySubject
            {
                /// @brief The adapter under test.
                MemoryFileRemover remover;

                /** @brief Puts a file at @p path. */
                void create(const std::string& path) { remover.add(path); }

                /** @brief True when a file is at @p path. */
                [[nodiscard]] bool exists(const std::string& path) const { return remover.has(path); }
            };

            /** @brief The disk adapter and the files it deletes. */
            struct DiskSubject
            {
                /// @brief The adapter under test.
                IOFiles::DiskFileRemover remover;

                /** @brief Puts a file at @p path. */
                void create(const std::string& path) { std::ofstream(path, std::ios::binary) << "partial"; }

                /** @brief True when a file is at @p path. */
                [[nodiscard]] bool exists(const std::string& path) const { return std::filesystem::exists(path); }
            };

            /** @brief A path in the temporary folder that no file holds. */
            [[nodiscard]] std::string free_path()
            {
                const std::filesystem::path path =
                    std::filesystem::temp_directory_path() / "sushiblas_file_remover.bin";
                std::filesystem::remove(path);
                return path.string();
            }
        } // namespace

        /** @brief Runs the FileRemover contract against the in-memory adapter. */
        template<typename Subject>
        class Small_FileRemoverContract : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        /** @brief Runs the FileRemover contract against the disk adapter. */
        template<typename Subject>
        class Medium_FileRemoverContract : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        using MemorySubjects = ::testing::Types<MemorySubject>;
        using DiskSubjects = ::testing::Types<DiskSubject>;
        SB_TYPED_TEST_SUITE(Small_FileRemoverContract, MemorySubjects);
        SB_TYPED_TEST_SUITE(Medium_FileRemoverContract, DiskSubjects);

        TYPED_TEST(Small_FileRemoverContract, ARemovedFileIsGone)
        {
            const std::string path = free_path();
            this->subject.create(path);

            EXPECT_TRUE(this->subject.remover.remove(path));
            EXPECT_FALSE(this->subject.exists(path));
        }

        TYPED_TEST(Small_FileRemoverContract, ARemoveOfNoFileFails)
        {
            EXPECT_FALSE(this->subject.remover.remove(free_path()));
        }

        TYPED_TEST(Medium_FileRemoverContract, ARemovedFileIsGone)
        {
            const std::string path = free_path();
            this->subject.create(path);

            EXPECT_TRUE(this->subject.remover.remove(path));
            EXPECT_FALSE(this->subject.exists(path));
        }

        TYPED_TEST(Medium_FileRemoverContract, ARemoveOfNoFileFails)
        {
            EXPECT_FALSE(this->subject.remover.remove(free_path()));
        }
    } // namespace Test
} // namespace SushiBLAS
