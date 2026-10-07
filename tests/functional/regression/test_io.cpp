/**************************************************************************/
/* test_io.cpp                                                            */
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
 * @file test_io.cpp
 * @brief Pins that IO refuses a file it cannot read whole and reaches device storage by copy.
 * @author Mustafa Garip
 */

#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Supplies an engine and temporary files that are removed after each test. */
        class Regression_IO : public SushiBLASTest
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

        TEST_F(Regression_IO, LoadRejectsATruncatedPayloadAndKeepsTheTensor)
        {
            const std::string path = temp_path("truncated_payload.sushi");
            auto source = create_initialized_tensor<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
            SushiBLAS::IO(*engine).save(source, path);
            drop_tail(path, 2 * sizeof(float));

            const std::vector<float> before{7.0f, 7.0f, 7.0f, 7.0f};
            auto target = create_initialized_tensor<float>({4}, before);
            EXPECT_THROW(SushiBLAS::IO(*engine).load(target, path), std::runtime_error);
            EXPECT_TENSOR_NEAR(target, before, 0.0);
        }

        TEST_F(Regression_IO, LoadRejectsAnotherDtype)
        {
            const std::string path = temp_path("dtype.sushi");
            auto source = create_initialized_tensor<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
            SushiBLAS::IO(*engine).save(source, path);

            auto target = engine->create_tensor({4}, Core::DataType::INT32);
            EXPECT_THROW(SushiBLAS::IO(*engine).load(target, path), std::runtime_error);
        }

        TEST_F(Regression_IO, LoadRejectsAnotherLayout)
        {
            const std::string path = temp_path("layout.sushi");
            {
                auto source = create_initialized_tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6});
                SushiBLAS::IO(*engine).save(source, path);
            }

            reinit_engine(Core::Layout::COLUMN_MAJOR);
            auto target = engine->create_tensor({2, 3}, Core::DataType::FLOAT32);
            EXPECT_THROW(SushiBLAS::IO(*engine).load(target, path), std::runtime_error);
        }

        TEST_F(Regression_IO, LoadRejectsAnotherFormatVersion)
        {
            // The version is the 32-bit field after the six magic bytes and two bytes of padding.
            constexpr size_t K_VERSION_OFFSET = 8;

            const std::string path = temp_path("version.sushi");
            auto source = create_initialized_tensor<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
            SushiBLAS::IO(*engine).save(source, path);

            std::vector<char> bytes = read_bytes(path);
            ASSERT_GT(bytes.size(), K_VERSION_OFFSET);
            ASSERT_EQ(bytes[K_VERSION_OFFSET], 1) << "The file does not hold version 1 where expected.";
            bytes[K_VERSION_OFFSET] = 2;
            write_bytes(path, bytes);

            auto target = engine->create_tensor({4}, Core::DataType::FLOAT32);
            EXPECT_THROW(SushiBLAS::IO(*engine).load(target, path), std::runtime_error);
        }

        TEST_F(Regression_IO, LoadBinRejectsAShortFileAndKeepsTheTensor)
        {
            const std::string path = temp_path("short.bin");
            auto source = create_initialized_tensor<float>({2}, {1.0f, 2.0f});
            SushiBLAS::IO(*engine).save_bin(source, path);

            const std::vector<float> before{7.0f, 7.0f, 7.0f, 7.0f};
            auto target = create_initialized_tensor<float>({4}, before);
            EXPECT_THROW(SushiBLAS::IO(*engine).load_bin(target, path), std::runtime_error);
            EXPECT_TENSOR_NEAR(target, before, 0.0);
        }

        TEST_F(Regression_IO, LoadNpyRejectsATruncatedPayload)
        {
            const std::string path = temp_path("truncated_payload.npy");
            auto source = create_initialized_tensor<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
            SushiBLAS::IO(*engine).save_npy(source, path);
            drop_tail(path, sizeof(float));

            auto target = engine->create_tensor({4}, Core::DataType::FLOAT32);
            EXPECT_THROW(SushiBLAS::IO(*engine).load_npy(target, path), std::runtime_error);
        }

        TEST_F(Regression_IO, BinRoundTripsDeviceAllocatedStorage)
        {
            if (!TestEnvironment::get_context().get_queue().get_device().is_gpu())
                GTEST_SKIP() << "Device USM is host-readable on this device; "
                                "the test proves nothing here.";

            const auto device = SushiRuntime::API::Residency::Device;
            const std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};

            const std::string path = temp_path("device.bin");
            auto source = engine->create_tensor({4}, Core::DataType::FLOAT32, device);
            fill_tensor(source, values);
            SushiBLAS::IO(*engine).save_bin(source, path);

            auto target = engine->create_tensor({4}, Core::DataType::FLOAT32, device);
            fill_tensor(target, {0.0f, 0.0f, 0.0f, 0.0f});
            SushiBLAS::IO(*engine).load_bin(target, path);

            EXPECT_TENSOR_NEAR(target, values, 0.0);
        }
    } // namespace Test
} // namespace SushiBLAS
