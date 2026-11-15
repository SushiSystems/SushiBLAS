/**************************************************************************/
/* test_planned_block.cpp                                                 */
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
 * @file test_planned_block.cpp
 * @brief Tests that PlannedBlock holds a planned runtime buffer with a handle, no host address
 *        and host copies through the buffer's ranges.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/planned_block.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Holds the bytes every planned block of these tests asks for.
        constexpr std::size_t K_PLANNED_BYTES = 256;

        /** @brief Returns n bytes counting up from first. */
        static std::vector<std::byte> counting_bytes(std::size_t n, unsigned first)
        {
            std::vector<std::byte> bytes(n);
            for (std::size_t i = 0; i < n; ++i)
                bytes[i] = static_cast<std::byte>((first + i) & 0xFFu);
            return bytes;
        }

        /** @brief Writes bytes to a file under the temporary folder and removes it on destruction. */
        class TemporaryFile final
        {
            public:
                /** @brief Writes bytes to a file named name. */
                TemporaryFile(const std::string& name, const std::vector<std::byte>& bytes)
                    : path_(::testing::TempDir() + name)
                {
                    std::ofstream out(path_, std::ios::binary | std::ios::trunc);
                    out.write(reinterpret_cast<const char*>(bytes.data()),
                              static_cast<std::streamsize>(bytes.size()));
                }

                TemporaryFile(const TemporaryFile&) = delete;
                TemporaryFile& operator=(const TemporaryFile&) = delete;

                /** @brief Removes the file. */
                ~TemporaryFile() { std::remove(path_.c_str()); }

                /** @brief Returns the path of the file. */
                [[nodiscard]] const std::string& path() const noexcept { return path_; }

            private:
                /// @brief Holds the path of the file.
                std::string path_;
        };

        TEST(Medium_PlannedBlock, HasAHandleAndNoHostAddress)
        {
            const PlannedBlock block(TestEnvironment::runtime(), K_PLANNED_BYTES);

            EXPECT_FALSE(block.global_handle().is_null());
            EXPECT_EQ(block.data(), nullptr);
            EXPECT_EQ(block.queue(), nullptr);
            EXPECT_EQ(block.residency(), SushiRuntime::Residency::Planned);
            EXPECT_EQ(block.size_bytes(), K_PLANNED_BYTES);
        }

        TEST(Medium_PlannedBlock, ReportsNoDevice)
        {
            const PlannedBlock block(TestEnvironment::runtime(), K_PLANNED_BYTES);

            EXPECT_EQ(block.device_index(), std::nullopt);
        }

        TEST(Medium_PlannedBlock, TwoLiveBlocksReportTwoHandles)
        {
            const PlannedBlock first(TestEnvironment::runtime(), K_PLANNED_BYTES);
            const PlannedBlock second(TestEnvironment::runtime(), K_PLANNED_BYTES);

            EXPECT_NE(first.global_handle(), second.global_handle());
        }

        TEST(Medium_PlannedBlock, ReadsBackWhatTheHostWrote)
        {
            const PlannedBlock block(TestEnvironment::runtime(), K_PLANNED_BYTES);
            const std::vector<std::byte> written = counting_bytes(K_PLANNED_BYTES, 3);

            block.copy_from_host(written.data(), 0, written.size());
            std::vector<std::byte> read(K_PLANNED_BYTES);
            block.copy_to_host(read.data(), 0, read.size());

            EXPECT_EQ(read, written);
        }

        TEST(Medium_PlannedBlock, ReadsAWindowAtAnOffset)
        {
            const PlannedBlock block(TestEnvironment::runtime(), K_PLANNED_BYTES);
            const std::vector<std::byte> written = counting_bytes(K_PLANNED_BYTES, 0);
            block.copy_from_host(written.data(), 0, written.size());

            std::vector<std::byte> window(16);
            block.copy_to_host(window.data(), 32, window.size());

            EXPECT_EQ(window, counting_bytes(16, 32));
        }

        TEST(Medium_PlannedBlock, RefusesARangeOutsideTheBlock)
        {
            const PlannedBlock block(TestEnvironment::runtime(), K_PLANNED_BYTES);
            std::vector<std::byte> read(16);

            EXPECT_THROW(block.copy_to_host(read.data(), K_PLANNED_BYTES - 8, read.size()),
                         std::runtime_error);
        }

        TEST(Medium_PlannedBlock, AFileBlockHoldsItsFileRange)
        {
            const std::vector<std::byte> contents = counting_bytes(K_PLANNED_BYTES, 0);
            const TemporaryFile file("sushiblas_planned_block_range.bin", contents);
            const PlannedBlock block(TestEnvironment::runtime(), file.path(), 64, 32);

            std::vector<std::byte> read(32);
            block.copy_to_host(read.data(), 0, read.size());

            EXPECT_EQ(block.size_bytes(), 32u);
            EXPECT_EQ(read, counting_bytes(32, 64));
        }

        TEST(Medium_PlannedBlock, RefusesAFileShorterThanTheRange)
        {
            const TemporaryFile file("sushiblas_planned_block_short.bin", counting_bytes(16, 0));

            EXPECT_THROW(PlannedBlock(TestEnvironment::runtime(), file.path(), 0, 32),
                         std::runtime_error);
        }

        TEST(Medium_PlannedBlock, AStorageOverTheBlockReportsItsHandle)
        {
            auto block = std::make_unique<PlannedBlock>(TestEnvironment::runtime(), K_PLANNED_BYTES);
            const SushiRuntime::GlobalHandle expected = block->global_handle();
            const SushiRuntime::sushi_ptr<Storage> storage =
                SushiRuntime::make_sushi<Storage>(std::move(block));

            EXPECT_EQ(storage->global_handle(), expected);
            EXPECT_EQ(storage->allocation_base(), nullptr);
        }
    } // namespace Test
} // namespace SushiBLAS
