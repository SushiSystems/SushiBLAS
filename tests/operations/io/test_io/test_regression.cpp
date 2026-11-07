/**************************************************************************/
/* test_regression.cpp                                                    */
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
 * @file test_regression.cpp
 * @brief Pins that IO refuses a file it cannot read whole and reaches device storage by copy.
 * @author Mustafa Garip
 */

#include <cstdio>
#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Supplies an engine and temporary files that are removed after each test. */
        class Medium_IO : public SushiBLASTest
        {
            protected:
                /** @brief Removes every file the test asked for, then tears the engine down. */
                void TearDown() override
                {
                    for (const std::string& path : paths_)
                        std::remove(path.c_str());

                    SushiBLASTest::TearDown();
                }

                /** @brief Returns a path in the test temp directory and records it for removal. */
                std::string temp_path(const std::string& name)
                {
                    paths_.push_back(::testing::TempDir() + "sushiblas_regression_io_" + name);
                    return paths_.back();
                }

                /** @brief Returns the whole content of the file at @p path. */
                static std::vector<char> read_bytes(const std::string& path)
                {
                    std::ifstream ifs(path, std::ios::binary);
                    return std::vector<char>(std::istreambuf_iterator<char>(ifs),
                                             std::istreambuf_iterator<char>());
                }

                /** @brief Replaces the file at @p path with @p bytes. */
                static void write_bytes(const std::string& path, const std::vector<char>& bytes)
                {
                    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
                    ofs.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
                }

                /** @brief Cuts @p count bytes off the end of the file at @p path. */
                static void drop_tail(const std::string& path, size_t count)
                {
                    std::vector<char> bytes = read_bytes(path);
                    ASSERT_GT(bytes.size(), count) << "The file is shorter than the tail to drop.";
                    bytes.resize(bytes.size() - count);
                    write_bytes(path, bytes);
                }

            private:
                /// @brief Lists the files this test created.
                std::vector<std::string> paths_;
        };

        namespace
        {
            /** @brief Names a file format IO loads whole or refuses. */
            enum class Format
            {
                Sushi, ///< The native .sushi format.
                Bin,   ///< Raw bytes with no header.
                Npy    ///< The NumPy .npy format.
            };

            /** @brief Returns the name a test instance carries for a format. */
            std::string format_name(Format format)
            {
                switch (format)
                {
                    case Format::Sushi: return "Sushi";
                    case Format::Bin:   return "Bin";
                    case Format::Npy:   return "Npy";
                }
                return "Unknown";
            }

            /** @brief Returns the name a test instance carries for a residency. */
            std::string memory_name(SushiRuntime::Residency residency)
            {
                return residency == SushiRuntime::Residency::Device ? "Device" : "Shared";
            }

            /** @brief One short file: its format, how much it lacks and where the target lives. */
            struct ShortFileRow
            {
                std::string name;
                Format format;
                size_t bytes_missing;
                SushiRuntime::Residency residency;
            };

            /** @brief Returns every format cut by one byte and by two elements, into each residency. */
            std::vector<ShortFileRow> short_file_rows()
            {
                std::vector<ShortFileRow> rows;
                for (const Format format : {Format::Sushi, Format::Bin, Format::Npy})
                {
                    for (const SushiRuntime::Residency residency :
                         {SushiRuntime::Residency::Shared, SushiRuntime::Residency::Device})
                    {
                        rows.push_back({format_name(format) + "OneByteShortInto" +
                                            memory_name(residency),
                                        format, 1, residency});
                        rows.push_back({format_name(format) + "TwoElementsShortInto" +
                                            memory_name(residency),
                                        format, 2 * sizeof(float), residency});
                    }
                }
                return rows;
            }

            /** @brief Returns the values a payload of count elements holds, the same on every call. */
            std::vector<float> chunk_values(size_t count)
            {
                std::vector<float> values(count);
                for (size_t i = 0; i < values.size(); ++i)
                    values[i] = static_cast<float>(i % 8191) - 4095.0f;
                return values;
            }
        } // namespace

        /** @brief Runs one short file of short_file_rows(). */
        class Medium_IOShortFile : public Medium_IO, public ::testing::WithParamInterface<ShortFileRow>
        {
            protected:
                /** @brief Saves four elements in the row's format and cuts the row's bytes off. */
                std::string short_file(const ShortFileRow& row)
                {
                    const std::string path = temp_path("short_" + row.name);
                    auto source = create_initialized_tensor<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
                    const SushiBLAS::IOStatus saved =
                        row.format == Format::Sushi ? SushiBLAS::IO().save(source, path)
                        : row.format == Format::Bin ? SushiBLAS::IO().save_bin(source, path)
                                                    : SushiBLAS::IO().save_npy(source, path);
                    EXPECT_EQ(saved, SushiBLAS::IOStatus::Ok);
                    drop_tail(path, row.bytes_missing);
                    return path;
                }

                /** @brief Loads the file at path into target in the row's format. */
                static SushiBLAS::IOStatus load(const ShortFileRow& row, Tensor& target,
                                                const std::string& path)
                {
                    return row.format == Format::Sushi ? SushiBLAS::IO().load(target, path)
                           : row.format == Format::Bin ? SushiBLAS::IO().load_bin(target, path)
                                                       : SushiBLAS::IO().load_npy(target, path);
                }
        };

        TEST_P(Medium_IOShortFile, IsRefusedAsTruncated)
        {
            const ShortFileRow row = GetParam();
            const std::string path = short_file(row);
            auto target = engine->create_tensor({4}, Core::DataType::Float32, row.residency);

            EXPECT_EQ(load(row, target, path), SushiBLAS::IOStatus::Truncated);
        }

        TEST_P(Medium_IOShortFile, LeavesTheTensorUntouched)
        {
            const ShortFileRow row = GetParam();
            const std::string path = short_file(row);
            const std::vector<float> sentinel{7.0f, 7.0f, 7.0f, 7.0f};
            auto target = engine->create_tensor({4}, Core::DataType::Float32, row.residency);
            fill_tensor(target, sentinel);

            static_cast<void>(load(row, target, path));

            expect_tensor_equal(target, sentinel);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_IOShortFile, ::testing::ValuesIn(short_file_rows()),
            [](const ::testing::TestParamInfo<ShortFileRow>& info) { return info.param.name; });

        /** @brief Skips a test on a device whose device memory the host can read. */
        class Medium_IODevice : public Medium_IO
        {
            protected:
                /** @brief Skips when device allocations are host-readable, as the test proves nothing there. */
                void SetUp() override
                {
                    Medium_IO::SetUp();
                    if (!TestEnvironment::queue().get_device().is_gpu())
                    {
                        GTEST_SKIP() << "Device USM is host-readable on this device; "
                                        "the test proves nothing here.";
                    }
                }
        };

        TEST_F(Medium_IODevice, BinRoundTripsDeviceAllocatedStorage)
        {
            const auto device = SushiRuntime::Residency::Device;
            const std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};

            const std::string path = temp_path("device.bin");
            auto source = engine->create_tensor({4}, Core::DataType::Float32, device);
            fill_tensor(source, values);
            ASSERT_EQ(SushiBLAS::IO().save_bin(source, path), SushiBLAS::IOStatus::Ok);

            auto target = engine->create_tensor({4}, Core::DataType::Float32, device);
            fill_tensor(target, {0.0f, 0.0f, 0.0f, 0.0f});
            ASSERT_EQ(SushiBLAS::IO().load_bin(target, path), SushiBLAS::IOStatus::Ok);

            expect_tensor_equal(target, values);
        }

        /** @brief Runs the chunked payload in one residency. */
        class Medium_IOChunks : public Medium_IO,
                                public ::testing::WithParamInterface<SushiRuntime::Residency>
        {

        };

        /**
         * @brief Tests that 3 x 2^20 FLOAT32 elements, 12 MiB and three of the 4 MiB chunks IO
         *        stages, save and load back unchanged.
         */
        TEST_P(Medium_IOChunks, ALargePayloadRoundTrips)
        {
            constexpr int64_t K_ELEMENTS = int64_t{3} << 20;
            const SushiRuntime::Residency residency = GetParam();
            const std::vector<float> values = chunk_values(static_cast<size_t>(K_ELEMENTS));
            const std::string path = temp_path("chunks_" + memory_name(residency) + ".bin");

            auto source = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32, residency);
            SushiBLAS::IO().from_vector(source, values);
            ASSERT_EQ(SushiBLAS::IO().save_bin(source, path), SushiBLAS::IOStatus::Ok);

            auto target = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32, residency);
            ASSERT_EQ(SushiBLAS::IO().load_bin(target, path), SushiBLAS::IOStatus::Ok);

            EXPECT_TRUE(SushiBLAS::IO().to_vector<float>(target) == values)
                << "The loaded elements differ from the saved ones.";
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_IOChunks,
            ::testing::Values(SushiRuntime::Residency::Device, SushiRuntime::Residency::Shared),
            [](const ::testing::TestParamInfo<SushiRuntime::Residency>& info)
            { return memory_name(info.param); });

        TEST_F(Medium_IO, ALongerFileStillLoadsItsPrefix)
        {
            const std::string path = temp_path("longer.bin");
            auto source = create_initialized_tensor<float>({6},
                                                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
            ASSERT_EQ(SushiBLAS::IO().save_bin(source, path), SushiBLAS::IOStatus::Ok);

            auto target = create_initialized_tensor<float>({4}, {7.0f, 7.0f, 7.0f, 7.0f});
            ASSERT_EQ(SushiBLAS::IO().load_bin(target, path), SushiBLAS::IOStatus::Ok);

            const std::vector<float> prefix{1.0f, 2.0f, 3.0f, 4.0f};
            expect_tensor_equal(target, prefix);
        }
    } // namespace Test
} // namespace SushiBLAS
