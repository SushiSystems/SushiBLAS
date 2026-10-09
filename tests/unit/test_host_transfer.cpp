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
 * @brief Tests IO's host transfer calls, load ordering, the .npy reader and the text of a tensor.
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
#include "../common/test_utils.hpp"

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
                    SushiBLAS::IO(*engine).from_vector(source, K_VALUES);
                    EXPECT_EQ(SushiBLAS::IO(*engine).save_npy(source, path),
                              SushiBLAS::IoStatus::Ok);
                    return path;
                }

                /**
                 * @brief Returns the elements of a rank-2 view in row order, read by its strides.
                 * @param parent The elements of the view's storage, from storage offset 0.
                 */
                static std::vector<float> gather_view(const Tensor& view,
                                                      const std::vector<float>& parent)
                {
                    std::vector<float> out;
                    out.reserve(static_cast<size_t>(view.num_elements));
                    for (int64_t i = 0; i < view.shape[0]; ++i)
                        for (int64_t j = 0; j < view.shape[1]; ++j)
                            out.push_back(parent[static_cast<size_t>(
                                view.storage_offset + i * view.strides[0] + j * view.strides[1])]);
                    return out;
                }

                /** @brief Returns the text of a dense Shared tensor of the view's elements. */
                std::string dense_twin_text(const Tensor& view, const std::vector<float>& parent)
                {
                    auto twin = engine->create_tensor(
                        SushiRuntime::span<const int64_t>(view.shape.data(),
                                                          static_cast<size_t>(view.rank)));
                    SushiBLAS::IO(*engine).from_vector(twin, gather_view(view, parent));
                    return SushiBLAS::IO(*engine).to_string(twin);
                }

            private:
                /// @brief Lists the files this test created.
                std::vector<std::string> paths_;
        };

        TEST_F(Unit_HostTransfer, ToVectorReadsBackWhatFromVectorWrote)
        {
            auto t = engine->create_tensor({2, 3});
            SushiBLAS::IO(*engine).from_vector(t, K_VALUES);

            EXPECT_EQ(SushiBLAS::IO(*engine).to_vector<float>(t), K_VALUES);
        }

        TEST_F(Unit_HostTransfer, ReadReturnsWhatWriteStored)
        {
            auto t = engine->create_tensor({2, 3});
            SushiBLAS::IO(*engine).write(t, K_VALUES.data(), K_VALUES.size() * sizeof(float));

            std::vector<float> out(K_VALUES.size());
            SushiBLAS::IO(*engine).read(t, out.data(), out.size() * sizeof(float));
            EXPECT_EQ(out, K_VALUES);
        }

        TEST_F(Unit_HostTransfer, SizeMismatchThrows)
        {
            auto t = engine->create_tensor({2, 3});
            const std::vector<float> short_values{1.0f, 2.0f};
            std::vector<float> out(K_VALUES.size() + 1);

            EXPECT_THROW(
                SushiBLAS::IO(*engine).from_vector(t, short_values),
                std::invalid_argument);
            EXPECT_THROW(SushiBLAS::IO(*engine).write(t, out.data(), out.size() * sizeof(float)),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::IO(*engine).read(t, out.data(), out.size() * sizeof(float)),
                         std::invalid_argument);
        }

        TEST_F(Unit_HostTransfer, ElementTypeMismatchThrows)
        {
            auto t = engine->create_tensor({2, 3});

            EXPECT_THROW((void)SushiBLAS::IO(*engine).to_vector<double>(t), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::IO(*engine).from_vector(t, std::vector<int32_t>(6, 1)),
                         std::invalid_argument);
        }

        TEST_F(Unit_HostTransfer, ReadRunsThePendingWriteFirst)
        {
            auto source = engine->create_tensor({2, 3});
            auto t = engine->create_tensor({2, 3});
            SushiBLAS::IO(*engine).from_vector(source, K_VALUES);
            SushiBLAS::IO(*engine).from_vector(t, K_PENDING_VALUES);

            SushiBLAS::BLASOps(*engine).copy(source, t);

            EXPECT_EQ(SushiBLAS::IO(*engine).to_vector<float>(t), K_VALUES);
        }

        TEST_F(Unit_HostTransfer, LoadNpyRunsThePendingWriteBeforeLoading)
        {
            const std::string path = save_values_npy("ordering.npy");

            auto pending = engine->create_tensor({2, 3});
            auto t = engine->create_tensor({2, 3});
            SushiBLAS::IO(*engine).from_vector(pending, K_PENDING_VALUES);

            SushiBLAS::BLASOps(*engine).copy(pending, t);
            ASSERT_EQ(SushiBLAS::IO(*engine).load_npy(t, path), SushiBLAS::IoStatus::Ok);

            EXPECT_EQ(SushiBLAS::IO(*engine).to_vector<float>(t), K_VALUES);
        }

        TEST_F(Unit_HostTransfer, LoadBinRunsThePendingWriteBeforeLoading)
        {
            const std::string path = temp_path("ordering.bin");
            {
                auto source = engine->create_tensor({2, 3});
                SushiBLAS::IO(*engine).from_vector(source, K_VALUES);
                ASSERT_EQ(SushiBLAS::IO(*engine).save_bin(source, path), SushiBLAS::IoStatus::Ok);
            }

            auto pending = engine->create_tensor({2, 3});
            auto t = engine->create_tensor({2, 3});
            SushiBLAS::IO(*engine).from_vector(pending, K_PENDING_VALUES);

            SushiBLAS::BLASOps(*engine).copy(pending, t);
            ASSERT_EQ(SushiBLAS::IO(*engine).load_bin(t, path), SushiBLAS::IoStatus::Ok);

            EXPECT_EQ(SushiBLAS::IO(*engine).to_vector<float>(t), K_VALUES);
        }

        TEST_F(Unit_HostTransfer, NpyRoundTripKeepsTheValues)
        {
            const std::string path = save_values_npy("round_trip.npy");

            auto t = engine->create_tensor({2, 3});
            ASSERT_EQ(SushiBLAS::IO(*engine).load_npy(t, path), SushiBLAS::IoStatus::Ok);

            EXPECT_EQ(SushiBLAS::IO(*engine).to_vector<float>(t), K_VALUES);
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
            ASSERT_EQ(SushiBLAS::IO(*engine).load_npy(t, path), SushiBLAS::IoStatus::Ok);

            EXPECT_EQ(SushiBLAS::IO(*engine).to_vector<float>(t), K_VALUES);
        }

        TEST_F(Unit_HostTransfer, NpyWithAnotherDtypeIsADtypeMismatch)
        {
            const std::string path = save_values_npy("dtype.npy");
            std::vector<char> bytes = read_bytes(path);
            replace_text(bytes, "<f4", "<f8");
            write_bytes(path, bytes);

            auto t = engine->create_tensor({2, 3});
            EXPECT_EQ(SushiBLAS::IO(*engine).load_npy(t, path), SushiBLAS::IoStatus::DtypeMismatch);
        }

        TEST_F(Unit_HostTransfer, NpyWithAnotherShapeIsAShapeMismatch)
        {
            const std::string path = save_values_npy("shape.npy");
            std::vector<char> bytes = read_bytes(path);
            replace_text(bytes, "(2, 3)", "(3, 2)");
            write_bytes(path, bytes);

            auto t = engine->create_tensor({2, 3});
            EXPECT_EQ(SushiBLAS::IO(*engine).load_npy(t, path), SushiBLAS::IoStatus::ShapeMismatch);
        }

        TEST_F(Unit_HostTransfer, NpyWithVersionFourIsMalformed)
        {
            const std::string path = save_values_npy("version_four.npy");
            std::vector<char> bytes = read_bytes(path);
            ASSERT_GT(bytes.size(), K_NPY_MAJOR_OFFSET);
            ASSERT_EQ(bytes[K_NPY_MAJOR_OFFSET], 1)
                << "The file does not hold version 1 where expected.";
            bytes[K_NPY_MAJOR_OFFSET] = 4;
            write_bytes(path, bytes);

            auto t = engine->create_tensor({2, 3});
            EXPECT_EQ(SushiBLAS::IO(*engine).load_npy(t, path), SushiBLAS::IoStatus::MalformedFile);
        }

        TEST_F(Unit_HostTransfer, NpyWithoutAShapeIsMalformed)
        {
            const std::string path = save_values_npy("no_shape.npy");
            std::vector<char> bytes = read_bytes(path);
            replace_text(bytes, "'shape'", "'shapf'");
            write_bytes(path, bytes);

            auto t = engine->create_tensor({2, 3});
            EXPECT_EQ(SushiBLAS::IO(*engine).load_npy(t, path), SushiBLAS::IoStatus::MalformedFile);
        }

        TEST_F(Unit_HostTransfer, NpyWithNumpyPaddingLoads)
        {
            const std::string path = temp_path("numpy_padding.npy");
            write_npy_v1(path, "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }");

            auto t = engine->create_tensor({2, 3});
            ASSERT_EQ(SushiBLAS::IO(*engine).load_npy(t, path), SushiBLAS::IoStatus::Ok);

            EXPECT_EQ(SushiBLAS::IO(*engine).to_vector<float>(t), K_VALUES);
        }

        TEST_F(Unit_HostTransfer, NpyWithoutCommasIsMalformed)
        {
            const std::string path = temp_path("no_commas.npy");
            write_npy_v1(path, "{'descr': '<f4' 'fortran_order': False 'shape': (2, 3) }");

            auto t = engine->create_tensor({2, 3});
            EXPECT_EQ(SushiBLAS::IO(*engine).load_npy(t, path), SushiBLAS::IoStatus::MalformedFile);
        }

        TEST_F(Unit_HostTransfer, NpyWithADuplicateKeyIsMalformed)
        {
            const std::string path = temp_path("duplicate_key.npy");
            write_npy_v1(path, "{'descr': '<f4', 'descr': '<f4', 'fortran_order': False, "
                               "'shape': (2, 3), }");

            auto t = engine->create_tensor({2, 3});
            EXPECT_EQ(SushiBLAS::IO(*engine).load_npy(t, path), SushiBLAS::IoStatus::MalformedFile);
        }

        TEST_F(Unit_HostTransfer, LoadNpyRejectsAColumnMajorRankThreeTensor)
        {
            reinit_engine(Core::Layout::ColumnMajor);
            auto t = engine->create_tensor({2, 2, 2});

            EXPECT_THROW((void)SushiBLAS::IO(*engine).load_npy(t, temp_path("column_major.npy")),
                         std::invalid_argument);
        }

        TEST_F(Unit_HostTransfer, ToStringOfADeviceTensorMatchesItsSharedTwin)
        {
            // With the default three edge items, the second shape of each rank is elided.
            const std::vector<std::vector<int64_t>> shapes{
                {5}, {20}, {2, 3}, {8, 9}, {2, 2, 3}, {7, 8, 9}};

            for (const std::vector<int64_t>& dims : shapes)
            {
                SCOPED_TRACE("rank " + std::to_string(dims.size()) + ", first dimension " +
                             std::to_string(dims.front()));
                const SushiRuntime::span<const int64_t> span(dims.data(), dims.size());
                auto shared = engine->create_tensor(span, Core::DataType::Float32,
                                                    SushiRuntime::API::Residency::Shared);
                auto device = engine->create_tensor(span, Core::DataType::Float32,
                                                    SushiRuntime::API::Residency::Device);

                std::vector<float> values(static_cast<size_t>(shared.num_elements));
                for (size_t i = 0; i < values.size(); ++i)
                    values[i] = 0.5f * static_cast<float>(i) - 3.0f;
                SushiBLAS::IO(*engine).from_vector(shared, values);
                SushiBLAS::IO(*engine).from_vector(device, values);

                EXPECT_EQ(SushiBLAS::IO(*engine).to_string(device),
                          SushiBLAS::IO(*engine).to_string(shared));
            }
        }

        TEST_F(Unit_HostTransfer, ToStringReadsAStridedViewThroughItsStrides)
        {
            const std::vector<float> values{0.0f, 1.0f, 2.0f, 3.0f, 4.0f,  5.0f,
                                            6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
            auto device = engine->create_tensor({3, 4}, Core::DataType::Float32,
                                                SushiRuntime::API::Residency::Device);
            SushiBLAS::IO(*engine).from_vector(device, values);

            const Tensor transposed = device.transpose(0, 1);
            EXPECT_EQ(SushiBLAS::IO(*engine).to_string(transposed),
                      dense_twin_text(transposed, values));

            const Tensor sliced = device.slice(1, 1, 3);
            EXPECT_EQ(SushiBLAS::IO(*engine).to_string(sliced), dense_twin_text(sliced, values));
        }

        TEST_F(Unit_HostTransfer, ToStringOfAnElidedTensorIsUnchanged)
        {
            std::vector<float> values(100);
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = static_cast<float>(i);

            auto t = engine->create_tensor({100});
            SushiBLAS::IO(*engine).from_vector(t, values);

            EXPECT_EQ(SushiBLAS::IO(*engine).to_string(t),
                      "Tensor(shape=[100], dtype=f32):\n"
                      "[0.0000, 1.0000, 2.0000, ... 97.0000, 98.0000, 99.0000]");
            EXPECT_EQ(SushiBLAS::IO(*engine).to_string(t, 4, 0),
                      "Tensor(shape=[100], dtype=f32):\n[... ]");
        }

        TEST_F(Unit_HostTransfer, ToStringOfAnEmptyTensorPrintsAnEmptyList)
        {
            auto t = engine->create_tensor({0});
            ASSERT_FALSE(t.storage) << "A zero-element tensor is expected to have no storage.";

            EXPECT_EQ(SushiBLAS::IO(*engine).to_string(t), "Tensor(shape=[0], dtype=f32):\n[]");
        }
    } // namespace Test
} // namespace SushiBLAS
