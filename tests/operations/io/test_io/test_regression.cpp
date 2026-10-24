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

        TEST_F(Medium_IO, LoadRejectsATruncatedPayloadAndKeepsTheTensor)
        {
            const std::string path = temp_path("truncated_payload.sushi");
            auto source = create_initialized_tensor<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
            ASSERT_EQ(SushiBLAS::IO().save(source, path), SushiBLAS::IOStatus::Ok);
            drop_tail(path, 2 * sizeof(float));

            const std::vector<float> before{7.0f, 7.0f, 7.0f, 7.0f};
            auto target = create_initialized_tensor<float>({4}, before);
            EXPECT_EQ(SushiBLAS::IO().load(target, path), SushiBLAS::IOStatus::Truncated);
            EXPECT_TENSOR_NEAR(target, before, 0.0);
        }

        TEST_F(Medium_IO, LoadRejectsAnotherDtype)
        {
            const std::string path = temp_path("dtype.sushi");
            auto source = create_initialized_tensor<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
            ASSERT_EQ(SushiBLAS::IO().save(source, path), SushiBLAS::IOStatus::Ok);

            auto target = engine->create_tensor({4}, Core::DataType::Int32);
            EXPECT_EQ(SushiBLAS::IO().load(target, path),
                      SushiBLAS::IOStatus::DtypeMismatch);
        }

        TEST_F(Medium_IO, LoadRejectsAnotherLayout)
        {
            const std::string path = temp_path("layout.sushi");
            {
                auto source = create_initialized_tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6});
                ASSERT_EQ(SushiBLAS::IO().save(source, path), SushiBLAS::IOStatus::Ok);
            }

            reinit_engine(Core::Layout::ColumnMajor);
            auto target = engine->create_tensor({2, 3}, Core::DataType::Float32);
            EXPECT_EQ(SushiBLAS::IO().load(target, path),
                      SushiBLAS::IOStatus::LayoutMismatch);
        }

        TEST_F(Medium_IO, LoadRejectsAnotherFormatVersion)
        {
            // The version is the 32-bit field after the six magic bytes and two bytes of padding.
            constexpr size_t K_VERSION_OFFSET = 8;

            const std::string path = temp_path("version.sushi");
            auto source = create_initialized_tensor<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
            ASSERT_EQ(SushiBLAS::IO().save(source, path), SushiBLAS::IOStatus::Ok);

            std::vector<char> bytes = read_bytes(path);
            ASSERT_GT(bytes.size(), K_VERSION_OFFSET);
            ASSERT_EQ(bytes[K_VERSION_OFFSET], 1) << "The file does not hold version 1 where expected.";
            bytes[K_VERSION_OFFSET] = 2;
            write_bytes(path, bytes);

            auto target = engine->create_tensor({4}, Core::DataType::Float32);
            EXPECT_EQ(SushiBLAS::IO().load(target, path),
                      SushiBLAS::IOStatus::MalformedFile);
        }

        TEST_F(Medium_IO, LoadBinRejectsAShortFileAndKeepsTheTensor)
        {
            const std::string path = temp_path("short.bin");
            auto source = create_initialized_tensor<float>({2}, {1.0f, 2.0f});
            ASSERT_EQ(SushiBLAS::IO().save_bin(source, path), SushiBLAS::IOStatus::Ok);

            const std::vector<float> before{7.0f, 7.0f, 7.0f, 7.0f};
            auto target = create_initialized_tensor<float>({4}, before);
            EXPECT_EQ(SushiBLAS::IO().load_bin(target, path),
                      SushiBLAS::IOStatus::Truncated);
            EXPECT_TENSOR_NEAR(target, before, 0.0);
        }

        TEST_F(Medium_IO, LoadNpyRejectsATruncatedPayload)
        {
            const std::string path = temp_path("truncated_payload.npy");
            auto source = create_initialized_tensor<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
            ASSERT_EQ(SushiBLAS::IO().save_npy(source, path), SushiBLAS::IOStatus::Ok);
            drop_tail(path, sizeof(float));

            auto target = engine->create_tensor({4}, Core::DataType::Float32);
            EXPECT_EQ(SushiBLAS::IO().load_npy(target, path),
                      SushiBLAS::IOStatus::Truncated);
        }

        TEST_F(Medium_IO, BinRoundTripsDeviceAllocatedStorage)
        {
            if (!TestEnvironment::queue().get_device().is_gpu())
                GTEST_SKIP() << "Device USM is host-readable on this device; "
                                "the test proves nothing here.";

            const auto device = SushiRuntime::Residency::Device;
            const std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};

            const std::string path = temp_path("device.bin");
            auto source = engine->create_tensor({4}, Core::DataType::Float32, device);
            fill_tensor(source, values);
            ASSERT_EQ(SushiBLAS::IO().save_bin(source, path), SushiBLAS::IOStatus::Ok);

            auto target = engine->create_tensor({4}, Core::DataType::Float32, device);
            fill_tensor(target, {0.0f, 0.0f, 0.0f, 0.0f});
            ASSERT_EQ(SushiBLAS::IO().load_bin(target, path), SushiBLAS::IOStatus::Ok);

            EXPECT_TENSOR_NEAR(target, values, 0.0);
        }

        TEST_F(Medium_IO, AShortFileLeavesTheTensorUntouched)
        {
            const std::string bin_path = temp_path("one_byte_short.bin");
            const std::string sushi_path = temp_path("one_byte_short.sushi");
            const std::string npy_path = temp_path("one_byte_short.npy");

            auto source = create_initialized_tensor<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
            ASSERT_EQ(SushiBLAS::IO().save_bin(source, bin_path), SushiBLAS::IOStatus::Ok);
            ASSERT_EQ(SushiBLAS::IO().save(source, sushi_path), SushiBLAS::IOStatus::Ok);
            ASSERT_EQ(SushiBLAS::IO().save_npy(source, npy_path), SushiBLAS::IOStatus::Ok);
            drop_tail(bin_path, 1);
            drop_tail(sushi_path, 1);
            drop_tail(npy_path, 1);

            const std::vector<float> sentinel{7.0f, 7.0f, 7.0f, 7.0f};
            for (const auto residency : {SushiRuntime::Residency::Shared,
                                         SushiRuntime::Residency::Device})
            {
                SCOPED_TRACE(residency == SushiRuntime::Residency::Device ? "Device"
                                                                               : "Shared");
                auto target = engine->create_tensor({4}, Core::DataType::Float32, residency);
                fill_tensor(target, sentinel);

                EXPECT_EQ(SushiBLAS::IO().load_bin(target, bin_path),
                          SushiBLAS::IOStatus::Truncated);
                EXPECT_TENSOR_NEAR(target, sentinel, 0.0);

                EXPECT_EQ(SushiBLAS::IO().load(target, sushi_path),
                          SushiBLAS::IOStatus::Truncated);
                EXPECT_TENSOR_NEAR(target, sentinel, 0.0);

                EXPECT_EQ(SushiBLAS::IO().load_npy(target, npy_path),
                          SushiBLAS::IOStatus::Truncated);
                EXPECT_TENSOR_NEAR(target, sentinel, 0.0);
            }
        }

        TEST_F(Medium_IO, APayloadLargerThanOneChunkRoundTrips)
        {
            // 3 x 2^20 FLOAT32 elements are 12 MiB, three of the 4 MiB chunks IO stages.
            constexpr int64_t K_ELEMENTS = int64_t{3} << 20;

            std::vector<float> values(static_cast<size_t>(K_ELEMENTS));
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = static_cast<float>(i % 8191) - 4095.0f;

            for (const auto residency : {SushiRuntime::Residency::Device,
                                         SushiRuntime::Residency::Shared})
            {
                const bool on_device = residency == SushiRuntime::Residency::Device;
                SCOPED_TRACE(on_device ? "Device" : "Shared");
                const std::string path =
                    temp_path(on_device ? "chunks_device.bin" : "chunks_shared.bin");

                auto source = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32,
                                                    residency);
                SushiBLAS::IO().from_vector(source, values);
                ASSERT_EQ(SushiBLAS::IO().save_bin(source, path), SushiBLAS::IOStatus::Ok);

                auto target = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32,
                                                    residency);
                ASSERT_EQ(SushiBLAS::IO().load_bin(target, path), SushiBLAS::IOStatus::Ok);

                EXPECT_TRUE(SushiBLAS::IO().to_vector<float>(target) == values)
                    << "The loaded elements differ from the saved ones.";
            }
        }

        TEST_F(Medium_IO, ALongerFileStillLoadsItsPrefix)
        {
            const std::string path = temp_path("longer.bin");
            auto source = create_initialized_tensor<float>({6},
                                                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
            ASSERT_EQ(SushiBLAS::IO().save_bin(source, path), SushiBLAS::IOStatus::Ok);

            auto target = create_initialized_tensor<float>({4}, {7.0f, 7.0f, 7.0f, 7.0f});
            ASSERT_EQ(SushiBLAS::IO().load_bin(target, path), SushiBLAS::IOStatus::Ok);

            const std::vector<float> prefix{1.0f, 2.0f, 3.0f, 4.0f};
            EXPECT_TENSOR_NEAR(target, prefix, 0.0);
        }
    } // namespace Test
} // namespace SushiBLAS
