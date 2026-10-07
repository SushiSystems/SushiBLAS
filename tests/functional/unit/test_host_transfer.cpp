/**************************************************************************/
/* test_host_transfer.cpp                                                 */
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
 * @file test_host_transfer.cpp
 * @brief Tests IO's host transfer calls, load ordering against pending writes and the .npy reader.
 * @author Mustafa Garip
 */

#include <cstdio>
#include <string>
#include <cstddef>
#include <algorithm>
#include <vector>
#include <cstdint>
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
        /** @brief Holds six FLOAT32 values written to and read back from a 2x3 tensor. */
        const std::vector<float> K_VALUES{1.5f, -2.0f, 3.25f, 0.0f, 4.0f, -5.5f};

        /** @brief Holds the values a pending copy would leave if a load ran before it. */
        const std::vector<float> K_PENDING_VALUES{7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f};

        /** @brief Holds the byte offset of the major version in a .npy file. */
        constexpr size_t K_NPY_MAJOR_OFFSET = 6;

        /** @brief Holds the byte offset of the version 1.0 header length in a .npy file. */
        constexpr size_t K_NPY_V1_LENGTH_OFFSET = 8;

        /** @brief Holds the byte offset of the version 1.0 header dictionary in a .npy file. */
        constexpr size_t K_NPY_V1_HEADER_OFFSET = 10;

        /** @brief Supplies an engine and temporary files that are removed after each test. */
        class Unit_HostTransfer : public SushiBLASTest
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
                    paths_.push_back(::testing::TempDir() + "sushiblas_host_transfer_" + name);
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

                /** @brief Replaces the first occurrence of @p from with @p to, of equal length. */
                static void replace_text(std::vector<char>& bytes, const std::string& from,
                                         const std::string& to)
                {
                    std::string text(bytes.begin(), bytes.end());
                    const size_t at = text.find(from);
                    ASSERT_NE(at, std::string::npos) << "'" << from << "' is not in the file.";
                    ASSERT_EQ(from.size(), to.size());
                    std::copy(to.begin(), to.end(),
                              bytes.begin() + static_cast<std::ptrdiff_t>(at));
                }

                /**
                 * @brief Writes a version 1.0 .npy file with @p dict as its header and K_VALUES.
                 * @param dict The header dictionary; space padding and a newline are appended.
                 */
                static void write_npy_v1(const std::string& path, const std::string& dict)
                {
                    std::string header = dict + " ";
                    while ((K_NPY_V1_HEADER_OFFSET + header.size() + 1) % 64 != 0)
                        header += " ";
                    header += "\n";

                    std::vector<char> bytes{'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0};
                    bytes.push_back(static_cast<char>(header.size() & 0xFF));
                    bytes.push_back(static_cast<char>((header.size() >> 8) & 0xFF));
                    bytes.insert(bytes.end(), header.begin(), header.end());

                    const char* payload = reinterpret_cast<const char*>(K_VALUES.data());
                    bytes.insert(bytes.end(), payload, payload + K_VALUES.size() * sizeof(float));
                    write_bytes(path, bytes);
                }

                /** @brief Saves a 2x3 FLOAT32 tensor of K_VALUES as .npy and returns the path. */
                std::string save_values_npy(const std::string& name)
                {
                    const std::string path = temp_path(name);
                    auto source = engine->create_tensor({2, 3});
                    engine->io().from_vector(source, K_VALUES);
                    engine->io().save_npy(source, path);
                    return path;
                }

            private:
                /// @brief Lists the files this test created.
                std::vector<std::string> paths_;
        };

        TEST_F(Unit_HostTransfer, ToVectorReadsBackWhatFromVectorWrote)
        {
            auto t = engine->create_tensor({2, 3});
            engine->io().from_vector(t, K_VALUES);

            EXPECT_EQ(engine->io().to_vector<float>(t), K_VALUES);
        }

        TEST_F(Unit_HostTransfer, ReadReturnsWhatWriteStored)
        {
            auto t = engine->create_tensor({2, 3});
            engine->io().write(t, K_VALUES.data(), K_VALUES.size() * sizeof(float));

            std::vector<float> out(K_VALUES.size());
            engine->io().read(t, out.data(), out.size() * sizeof(float));
            EXPECT_EQ(out, K_VALUES);
        }

        TEST_F(Unit_HostTransfer, SizeMismatchThrows)
        {
            auto t = engine->create_tensor({2, 3});
            const std::vector<float> short_values{1.0f, 2.0f};
            std::vector<float> out(K_VALUES.size() + 1);

            EXPECT_THROW(engine->io().from_vector(t, short_values), std::invalid_argument);
            EXPECT_THROW(engine->io().write(t, out.data(), out.size() * sizeof(float)),
                         std::invalid_argument);
            EXPECT_THROW(engine->io().read(t, out.data(), out.size() * sizeof(float)),
                         std::invalid_argument);
        }

        TEST_F(Unit_HostTransfer, ElementTypeMismatchThrows)
        {
            auto t = engine->create_tensor({2, 3});

            EXPECT_THROW((void)engine->io().to_vector<double>(t), std::invalid_argument);
            EXPECT_THROW(engine->io().from_vector(t, std::vector<int32_t>(6, 1)),
                         std::invalid_argument);
        }

        TEST_F(Unit_HostTransfer, ReadRunsThePendingWriteFirst)
        {
            auto source = engine->create_tensor({2, 3});
            auto t = engine->create_tensor({2, 3});
            engine->io().from_vector(source, K_VALUES);
            engine->io().from_vector(t, K_PENDING_VALUES);

            engine->blas().copy(source, t);

            EXPECT_EQ(engine->io().to_vector<float>(t), K_VALUES);
        }

        TEST_F(Unit_HostTransfer, LoadNpyRunsThePendingWriteBeforeLoading)
        {
            const std::string path = save_values_npy("ordering.npy");

            auto pending = engine->create_tensor({2, 3});
            auto t = engine->create_tensor({2, 3});
            engine->io().from_vector(pending, K_PENDING_VALUES);

            engine->blas().copy(pending, t);
            engine->io().load_npy(t, path);

            EXPECT_EQ(engine->io().to_vector<float>(t), K_VALUES);
        }

        TEST_F(Unit_HostTransfer, LoadBinRunsThePendingWriteBeforeLoading)
        {
            const std::string path = temp_path("ordering.bin");
            {
                auto source = engine->create_tensor({2, 3});
                engine->io().from_vector(source, K_VALUES);
                engine->io().save_bin(source, path);
            }

            auto pending = engine->create_tensor({2, 3});
            auto t = engine->create_tensor({2, 3});
            engine->io().from_vector(pending, K_PENDING_VALUES);

            engine->blas().copy(pending, t);
            engine->io().load_bin(t, path);

            EXPECT_EQ(engine->io().to_vector<float>(t), K_VALUES);
        }

        TEST_F(Unit_HostTransfer, NpyRoundTripKeepsTheValues)
        {
            const std::string path = save_values_npy("round_trip.npy");

            auto t = engine->create_tensor({2, 3});
            engine->io().load_npy(t, path);

            EXPECT_EQ(engine->io().to_vector<float>(t), K_VALUES);
        }

        TEST_F(Unit_HostTransfer, NpyVersionTwoHeaderLoads)
        {
            const std::string path = save_values_npy("version_two.npy");
            const std::vector<char> v1 = read_bytes(path);
            ASSERT_GT(v1.size(), K_NPY_V1_HEADER_OFFSET);

            const uint8_t low = static_cast<uint8_t>(v1[K_NPY_V1_LENGTH_OFFSET]);
            const uint8_t high = static_cast<uint8_t>(v1[K_NPY_V1_LENGTH_OFFSET + 1]);

            std::vector<char> v2(v1.begin(), v1.begin() + K_NPY_MAJOR_OFFSET);
            v2.push_back(2);
            v2.push_back(0);
            v2.push_back(static_cast<char>(low));
            v2.push_back(static_cast<char>(high));
            v2.push_back(0);
            v2.push_back(0);
            v2.insert(v2.end(), v1.begin() + K_NPY_V1_HEADER_OFFSET, v1.end());
            write_bytes(path, v2);

            auto t = engine->create_tensor({2, 3});
            engine->io().load_npy(t, path);

            EXPECT_EQ(engine->io().to_vector<float>(t), K_VALUES);
        }

        TEST_F(Unit_HostTransfer, NpyWithAnotherDtypeThrows)
        {
            const std::string path = save_values_npy("dtype.npy");
            std::vector<char> bytes = read_bytes(path);
            replace_text(bytes, "<f4", "<f8");
            write_bytes(path, bytes);

            auto t = engine->create_tensor({2, 3});
            EXPECT_THROW(engine->io().load_npy(t, path), std::runtime_error);
        }

        TEST_F(Unit_HostTransfer, NpyWithAnotherShapeThrows)
        {
            const std::string path = save_values_npy("shape.npy");
            std::vector<char> bytes = read_bytes(path);
            replace_text(bytes, "(2, 3)", "(3, 2)");
            write_bytes(path, bytes);

            auto t = engine->create_tensor({2, 3});
            EXPECT_THROW(engine->io().load_npy(t, path), std::runtime_error);
        }

        TEST_F(Unit_HostTransfer, NpyWithVersionFourThrows)
        {
            const std::string path = save_values_npy("version_four.npy");
            std::vector<char> bytes = read_bytes(path);
            ASSERT_GT(bytes.size(), K_NPY_MAJOR_OFFSET);
            ASSERT_EQ(bytes[K_NPY_MAJOR_OFFSET], 1)
                << "The file does not hold version 1 where expected.";
            bytes[K_NPY_MAJOR_OFFSET] = 4;
            write_bytes(path, bytes);

            auto t = engine->create_tensor({2, 3});
            EXPECT_THROW(engine->io().load_npy(t, path), std::runtime_error);
        }

        TEST_F(Unit_HostTransfer, NpyWithoutAShapeThrows)
        {
            const std::string path = save_values_npy("no_shape.npy");
            std::vector<char> bytes = read_bytes(path);
            replace_text(bytes, "'shape'", "'shapf'");
            write_bytes(path, bytes);

            auto t = engine->create_tensor({2, 3});
            EXPECT_THROW(engine->io().load_npy(t, path), std::runtime_error);
        }

        TEST_F(Unit_HostTransfer, NpyWithNumpyPaddingLoads)
        {
            const std::string path = temp_path("numpy_padding.npy");
            write_npy_v1(path, "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }");

            auto t = engine->create_tensor({2, 3});
            engine->io().load_npy(t, path);

            EXPECT_EQ(engine->io().to_vector<float>(t), K_VALUES);
        }

        TEST_F(Unit_HostTransfer, NpyWithoutCommasThrows)
        {
            const std::string path = temp_path("no_commas.npy");
            write_npy_v1(path, "{'descr': '<f4' 'fortran_order': False 'shape': (2, 3) }");

            auto t = engine->create_tensor({2, 3});
            EXPECT_THROW(engine->io().load_npy(t, path), std::runtime_error);
        }

        TEST_F(Unit_HostTransfer, NpyWithADuplicateKeyThrows)
        {
            const std::string path = temp_path("duplicate_key.npy");
            write_npy_v1(path, "{'descr': '<f4', 'descr': '<f4', 'fortran_order': False, "
                               "'shape': (2, 3), }");

            auto t = engine->create_tensor({2, 3});
            EXPECT_THROW(engine->io().load_npy(t, path), std::runtime_error);
        }

        TEST_F(Unit_HostTransfer, LoadNpyRejectsAColumnMajorRankThreeTensor)
        {
            reinit_engine(Core::Layout::COLUMN_MAJOR);
            auto t = engine->create_tensor({2, 2, 2});

            EXPECT_THROW(engine->io().load_npy(t, temp_path("column_major.npy")),
                         std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
