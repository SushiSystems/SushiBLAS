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
 * @file test_host_transfer.cpp
 * @brief Tests IO's host transfer calls, loads after a run, the .npy reader and the text of a
 *        tensor.
 * @author Mustafa Garip
 */

#include <cstdio>
#include <string>
#include <cstddef>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds six FLOAT32 values written to and read back from a 2x3 tensor. */
        const std::vector<float> K_VALUES{1.5f, -2.0f, 3.25f, 0.0f, 4.0f, -5.5f};

        /** @brief Holds the values the recorded copy writes before a load overwrites them. */
        const std::vector<float> K_PENDING_VALUES{7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f};

        namespace
        {
            /** @brief Names a file format IO loads after the caller's run. */
            enum class Loader
            {
                Npy, ///< The NumPy .npy format.
                Bin  ///< Raw bytes with no header.
            };

            /** @brief Returns count values that rise by one half from -3, the same on every call. */
            std::vector<float> ramp_values(size_t count)
            {
                std::vector<float> values(count);
                for (size_t i = 0; i < values.size(); ++i)
                    values[i] = 0.5f * static_cast<float>(i) - 3.0f;
                return values;
            }
        } // namespace

        /** @brief Supplies an engine and temporary files that are removed after each test. */
        class Medium_HostTransfer : public SushiBLASTest
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

                /** @brief Saves a 2x3 FLOAT32 tensor of K_VALUES as .npy and returns the path. */
                std::string save_values_npy(const std::string& name)
                {
                    const std::string path = temp_path(name);
                    auto source = engine->create_tensor({2, 3});
                    SushiBLAS::IO().from_vector(source, K_VALUES);
                    EXPECT_EQ(SushiBLAS::IO().save_npy(source, path),
                              SushiBLAS::IOStatus::Ok);
                    return path;
                }

                /** @brief Saves K_VALUES of a 2x3 tensor in the format and returns the path. */
                std::string save_values(Loader loader, const std::string& name)
                {
                    if (loader == Loader::Npy)
                        return save_values_npy(name + ".npy");

                    const std::string path = temp_path(name + ".bin");
                    auto source = engine->create_tensor({2, 3});
                    SushiBLAS::IO().from_vector(source, K_VALUES);
                    EXPECT_EQ(SushiBLAS::IO().save_bin(source, path), SushiBLAS::IOStatus::Ok);
                    return path;
                }

                /** @brief Loads the file at path into t in the format. */
                static SushiBLAS::IOStatus load(Loader loader, Tensor& t, const std::string& path)
                {
                    return loader == Loader::Npy ? SushiBLAS::IO().load_npy(t, path)
                                                 : SushiBLAS::IO().load_bin(t, path);
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
                    SushiBLAS::IO().from_vector(twin, gather_view(view, parent));
                    return SushiBLAS::IO().to_string(twin);
                }

            private:
                /// @brief Lists the files this test created.
                std::vector<std::string> paths_;
        };

        TEST_F(Medium_HostTransfer, ToVectorReadsBackWhatFromVectorWrote)
        {
            auto t = engine->create_tensor({2, 3});
            SushiBLAS::IO().from_vector(t, K_VALUES);

            EXPECT_EQ(SushiBLAS::IO().to_vector<float>(t), K_VALUES);
        }

        TEST_F(Medium_HostTransfer, ReadReturnsWhatWriteStored)
        {
            auto t = engine->create_tensor({2, 3});
            SushiBLAS::IO().write(t, K_VALUES.data(), K_VALUES.size() * sizeof(float));

            std::vector<float> out(K_VALUES.size());
            SushiBLAS::IO().read(t, out.data(), out.size() * sizeof(float));
            EXPECT_EQ(out, K_VALUES);
        }

        namespace
        {
            /** @brief Names a call that refuses a buffer of the wrong size. */
            enum class SizeCall
            {
                FromVector, ///< from_vector with fewer values than the tensor holds.
                Write,      ///< write with more bytes than the tensor holds.
                Read        ///< read with more bytes than the tensor holds.
            };

            /** @brief One call of the size table. */
            struct SizeRow
            {
                const char* name;
                SizeCall call;
            };

            /** @brief Returns the calls that take a size. */
            std::vector<SizeRow> size_rows()
            {
                return {{"FromVector", SizeCall::FromVector},
                        {"Write", SizeCall::Write},
                        {"Read", SizeCall::Read}};
            }

            /** @brief Calls IO with a buffer whose size is not the tensor's. */
            void call_with_a_wrong_size(SizeCall call, Tensor& t)
            {
                const std::vector<float> short_values{1.0f, 2.0f};
                std::vector<float> out(K_VALUES.size() + 1);
                switch (call)
                {
                    case SizeCall::FromVector:
                        SushiBLAS::IO().from_vector(t, short_values);
                        break;
                    case SizeCall::Write:
                        SushiBLAS::IO().write(t, out.data(), out.size() * sizeof(float));
                        break;
                    case SizeCall::Read:
                        SushiBLAS::IO().read(t, out.data(), out.size() * sizeof(float));
                        break;
                }
            }

            /** @brief Names a call that refuses an element type other than the tensor's. */
            enum class TypeCall
            {
                ToVectorOfDouble, ///< to_vector of double over a float tensor.
                FromVectorOfInt32 ///< from_vector of int32 into a float tensor.
            };

            /** @brief One call of the element type table. */
            struct TypeRow
            {
                const char* name;
                TypeCall call;
            };

            /** @brief Returns the calls that take an element type. */
            std::vector<TypeRow> type_rows()
            {
                return {{"ToVectorOfDouble", TypeCall::ToVectorOfDouble},
                        {"FromVectorOfInt32", TypeCall::FromVectorOfInt32}};
            }

            /** @brief Calls IO with an element type other than the tensor's. */
            void call_with_another_element_type(TypeCall call, Tensor& t)
            {
                switch (call)
                {
                    case TypeCall::ToVectorOfDouble:
                        (void)SushiBLAS::IO().to_vector<double>(t);
                        break;
                    case TypeCall::FromVectorOfInt32:
                        SushiBLAS::IO().from_vector(t, std::vector<int32_t>(6, 1));
                        break;
                }
            }

            /** @brief One loader of the table of loaders. */
            struct LoaderRow
            {
                const char* name;
                Loader loader;
            };

            /** @brief Returns the loaders. */
            std::vector<LoaderRow> loader_rows()
            {
                return {{"Npy", Loader::Npy}, {"Bin", Loader::Bin}};
            }

            /** @brief One shape the text of a device tensor is compared at. */
            struct ShapeRow
            {
                const char* name;
                std::vector<int64_t> dims;
            };

            /**
             * @brief Returns shapes of rank one to three; with the default three edge items the
             *        second shape of each rank is elided.
             */
            std::vector<ShapeRow> shape_rows()
            {
                return {{"RankOneShort", {5}},         {"RankOneElided", {20}},
                        {"RankTwoShort", {2, 3}},      {"RankTwoElided", {8, 9}},
                        {"RankThreeShort", {2, 2, 3}}, {"RankThreeElided", {7, 8, 9}}};
            }

            /** @brief Names a strided view of a 3x4 tensor. */
            enum class View
            {
                Transposed, ///< The tensor with its two dimensions swapped.
                Sliced      ///< Columns one and two of the tensor.
            };

            /** @brief One view of the table of views. */
            struct ViewRow
            {
                const char* name;
                View view;
            };

            /** @brief Returns the views. */
            std::vector<ViewRow> view_rows()
            {
                return {{"Transposed", View::Transposed}, {"Sliced", View::Sliced}};
            }

            /** @brief Returns the strided view of a 3x4 tensor the row names. */
            Tensor view_of(const Tensor& device, View view)
            {
                return view == View::Transposed ? device.transpose(0, 1) : device.slice(1, 1, 3);
            }
        } // namespace

        /** @brief Runs one call of size_rows(). */
        class Medium_HostTransferSizeMismatch : public Medium_HostTransfer,
                                                public ::testing::WithParamInterface<SizeRow>
        {

        };

        TEST_P(Medium_HostTransferSizeMismatch, IsRefused)
        {
            auto t = engine->create_tensor({2, 3});

            EXPECT_THROW(call_with_a_wrong_size(GetParam().call, t), std::invalid_argument);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_HostTransferSizeMismatch, ::testing::ValuesIn(size_rows()),
            [](const ::testing::TestParamInfo<SizeRow>& info)
            { return std::string(info.param.name); });

        /** @brief Runs one call of type_rows(). */
        class Medium_HostTransferElementTypeMismatch
            : public Medium_HostTransfer, public ::testing::WithParamInterface<TypeRow>
        {

        };

        TEST_P(Medium_HostTransferElementTypeMismatch, IsRefused)
        {
            auto t = engine->create_tensor({2, 3});

            EXPECT_THROW(call_with_another_element_type(GetParam().call, t),
                         std::invalid_argument);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_HostTransferElementTypeMismatch, ::testing::ValuesIn(type_rows()),
            [](const ::testing::TestParamInfo<TypeRow>& info)
            { return std::string(info.param.name); });

        TEST_F(Medium_HostTransfer, ReadSeesTheWriteTheCallerRan)
        {
            auto source = engine->create_tensor({2, 3});
            auto t = engine->create_tensor({2, 3});
            SushiBLAS::IO().from_vector(source, K_VALUES);
            SushiBLAS::IO().from_vector(t, K_PENDING_VALUES);

            SushiBLAS::BLASOps(recorder()).copy(source, t);
            recording->run().value();

            EXPECT_EQ(SushiBLAS::IO().to_vector<float>(t), K_VALUES);
        }

        /** @brief Runs one loader of loader_rows(). */
        class Medium_HostTransferLoadAfterTheRun : public Medium_HostTransfer,
                                                   public ::testing::WithParamInterface<LoaderRow>
        {

        };

        TEST_P(Medium_HostTransferLoadAfterTheRun, OverwritesTheWriteTheCallerRan)
        {
            const Loader loader = GetParam().loader;
            const std::string path = save_values(loader, "ordering");

            auto pending = engine->create_tensor({2, 3});
            auto t = engine->create_tensor({2, 3});
            SushiBLAS::IO().from_vector(pending, K_PENDING_VALUES);

            SushiBLAS::BLASOps(recorder()).copy(pending, t);
            recording->run().value();
            ASSERT_EQ(load(loader, t, path), SushiBLAS::IOStatus::Ok);

            EXPECT_EQ(SushiBLAS::IO().to_vector<float>(t), K_VALUES);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_HostTransferLoadAfterTheRun, ::testing::ValuesIn(loader_rows()),
            [](const ::testing::TestParamInfo<LoaderRow>& info)
            { return std::string(info.param.name); });

        TEST_F(Medium_HostTransfer, NpyRoundTripKeepsTheValues)
        {
            const std::string path = save_values_npy("round_trip.npy");

            auto t = engine->create_tensor({2, 3});
            ASSERT_EQ(SushiBLAS::IO().load_npy(t, path), SushiBLAS::IOStatus::Ok);

            EXPECT_EQ(SushiBLAS::IO().to_vector<float>(t), K_VALUES);
        }

        TEST_F(Medium_HostTransfer, LoadNpyRejectsAColumnMajorRankThreeTensor)
        {
            reinit_engine(Core::Layout::ColumnMajor);
            auto t = engine->create_tensor({2, 2, 2});

            EXPECT_THROW((void)SushiBLAS::IO().load_npy(t, temp_path("column_major.npy")),
                         std::invalid_argument);
        }

        /** @brief Runs one shape of shape_rows(). */
        class Medium_HostTransferToStringOfADeviceTensor
            : public Medium_HostTransfer, public ::testing::WithParamInterface<ShapeRow>
        {

        };

        TEST_P(Medium_HostTransferToStringOfADeviceTensor, MatchesItsSharedTwin)
        {
            const std::vector<int64_t>& dims = GetParam().dims;
            const SushiRuntime::span<const int64_t> span(dims.data(), dims.size());
            auto shared = engine->create_tensor(span, Core::DataType::Float32,
                                                SushiRuntime::Residency::Shared);
            auto device = engine->create_tensor(span, Core::DataType::Float32,
                                                SushiRuntime::Residency::Device);
            const std::vector<float> values =
                ramp_values(static_cast<size_t>(shared.num_elements));
            SushiBLAS::IO().from_vector(shared, values);
            SushiBLAS::IO().from_vector(device, values);

            EXPECT_EQ(SushiBLAS::IO().to_string(device), SushiBLAS::IO().to_string(shared));
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_HostTransferToStringOfADeviceTensor, ::testing::ValuesIn(shape_rows()),
            [](const ::testing::TestParamInfo<ShapeRow>& info)
            { return std::string(info.param.name); });

        /** @brief Runs one view of view_rows(). */
        class Medium_HostTransferToStringOfAStridedView
            : public Medium_HostTransfer, public ::testing::WithParamInterface<ViewRow>
        {

        };

        TEST_P(Medium_HostTransferToStringOfAStridedView, ReadsTheViewThroughItsStrides)
        {
            const std::vector<float> values = ramp_values(12);
            auto device = engine->create_tensor({3, 4}, Core::DataType::Float32,
                                                SushiRuntime::Residency::Device);
            SushiBLAS::IO().from_vector(device, values);

            const Tensor view = view_of(device, GetParam().view);

            EXPECT_EQ(SushiBLAS::IO().to_string(view), dense_twin_text(view, values));
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_HostTransferToStringOfAStridedView, ::testing::ValuesIn(view_rows()),
            [](const ::testing::TestParamInfo<ViewRow>& info)
            { return std::string(info.param.name); });

        /**
         * @brief Tests that the default of three edge items prints the first three and the last
         *        three of 100 values around an ellipsis: 0, 1, 2, then 97, 98, 99.
         */
        TEST_F(Medium_HostTransfer, ToStringKeepsThreeItemsAtEachEdgeOfALongRun)
        {
            std::vector<float> values(100);
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = static_cast<float>(i);
            auto t = engine->create_tensor({100});
            SushiBLAS::IO().from_vector(t, values);

            EXPECT_EQ(SushiBLAS::IO().to_string(t),
                      "Tensor(shape=[100], dtype=f32):\n"
                      "[0.0000, 1.0000, 2.0000, ... 97.0000, 98.0000, 99.0000]");
        }

        /**
         * @brief Tests that zero edge items leave only the ellipsis of a run of 100 values.
         */
        TEST_F(Medium_HostTransfer, ToStringWithZeroEdgeItemsPrintsOnlyTheEllipsis)
        {
            auto t = engine->create_tensor({100});
            SushiBLAS::IO().from_vector(t, std::vector<float>(100, 1.0f));

            EXPECT_EQ(SushiBLAS::IO().to_string(t, 4, 0),
                      "Tensor(shape=[100], dtype=f32):\n[... ]");
        }

        TEST_F(Medium_HostTransfer, ToStringOfAnEmptyTensorPrintsAnEmptyList)
        {
            auto t = engine->create_tensor({0});

            EXPECT_EQ(SushiBLAS::IO().to_string(t), "Tensor(shape=[0], dtype=f32):\n[]");
        }
    } // namespace Test
} // namespace SushiBLAS
