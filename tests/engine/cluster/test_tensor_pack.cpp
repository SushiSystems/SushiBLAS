/**************************************************************************/
/* test_tensor_pack.cpp                                                   */
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
 * @file test_tensor_pack.cpp
 * @brief Tests that TensorPack gathers tensors into its buffer and scatters them back unchanged.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor/runtime_block.hpp>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiBLAS/cluster/tensor_pack.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiBLAS/engine/runtime_scratch_source.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns the program-wide runtime every test of this file packs on. */
            SushiRuntime::Runtime& runtime()
            {
                return TestEnvironment::runtime();
            }

            /** @brief Returns a shared tensor of a type with count elements on one device. */
            Tensor shared_tensor(Core::DataType dtype, std::int64_t count,
                                 std::size_t device_index = 0)
            {
                const std::int64_t dims[] = {count};
                const std::size_t bytes =
                    static_cast<std::size_t>(count) * Core::bytes_per_element(dtype);
                return Tensor(make_runtime_storage(runtime(), bytes,
                                                   SushiRuntime::Residency::Shared, device_index),
                              SushiRuntime::span<const std::int64_t>(dims, 1), dtype);
            }

            /** @brief Returns the byte count of a tensor's elements. */
            std::size_t byte_count(const Tensor& tensor)
            {
                return static_cast<std::size_t>(tensor.num_elements) *
                       Core::bytes_per_element(tensor.dtype);
            }

            /** @brief Writes the bytes offset * 31 + salt into a tensor, wrapping at 256. */
            void fill_pattern(Tensor& tensor, std::size_t salt)
            {
                auto* bytes = static_cast<std::uint8_t*>(tensor.data());
                for (std::size_t offset = 0; offset < byte_count(tensor); ++offset)
                    bytes[offset] = static_cast<std::uint8_t>(offset * 31 + salt);
            }

            /** @brief Returns the layout of a list of tensors. */
            PackLayout layout_of(const std::vector<Tensor>& tensors)
            {
                return PackLayout::of(
                    SushiRuntime::span<const Tensor>(tensors.data(), tensors.size()));
            }

            /** @brief Returns true when two tensors hold the same bytes. */
            bool same_bytes(const Tensor& left, const Tensor& right)
            {
                return byte_count(left) == byte_count(right) &&
                       std::memcmp(left.data(), right.data(), byte_count(left)) == 0;
            }

            /** @brief Records a gather of the sources into the pack and runs it. */
            void gather(const TensorPack& pack, const std::vector<Tensor>& sources)
            {
                RuntimeScratchSource scratch_source(runtime());
                ScratchPool scratch(scratch_source);
                LateCountHold late_counts(runtime());
                auto graph = runtime().graph();
                graph.extend([&](SushiRuntime::Recorder& recorder)
                {
                    TaskRecorder tensors(recorder, scratch, late_counts);
                    (void)pack.record_gather(tensors, SushiRuntime::span<const Tensor>(
                                                          sources.data(), sources.size()));
                });
                graph.run().value();
            }

            /** @brief Records a gather then a scatter into the targets and runs it. */
            void gather_then_scatter(const TensorPack& pack, const std::vector<Tensor>& sources,
                                     const std::vector<Tensor>& targets)
            {
                RuntimeScratchSource scratch_source(runtime());
                ScratchPool scratch(scratch_source);
                LateCountHold late_counts(runtime());
                auto graph = runtime().graph();
                graph.extend([&](SushiRuntime::Recorder& recorder)
                {
                    TaskRecorder tensors(recorder, scratch, late_counts);
                    (void)pack.record_gather(tensors, SushiRuntime::span<const Tensor>(
                                                          sources.data(), sources.size()));
                    (void)pack.record_scatter(tensors, SushiRuntime::span<const Tensor>(
                                                           targets.data(), targets.size()));
                });
                graph.run().value();
            }

            /** @brief Gathers the sources into a pack that dies before this returns, and hands back its tensors. */
            std::vector<Tensor> entry_views_after_the_pack_dies(const std::vector<Tensor>& sources)
            {
                std::vector<Tensor> views;
                const TensorPack pack(runtime(), layout_of(sources), 0);
                gather(pack, sources);
                for (std::size_t index = 0; index < sources.size(); ++index)
                    views.push_back(pack.tensor(index));
                return views;
            }

            /** @brief Names the data type one segment tensor row checks. */
            struct SegmentRow
            {
                const char* name;     /**< The row name a failure shows. */
                Core::DataType dtype; /**< The type whose segment tensor is read. */
            };

            /** @brief Returns one row per data type. */
            std::vector<SegmentRow> segment_rows()
            {
                return {{"Half", Core::DataType::Half},
                        {"Float32", Core::DataType::Float32},
                        {"Float64", Core::DataType::Float64},
                        {"Complex32", Core::DataType::Complex32},
                        {"Complex64", Core::DataType::Complex64},
                        {"Int32", Core::DataType::Int32},
                        {"Int64", Core::DataType::Int64},
                        {"UInt32", Core::DataType::UInt32}};
            }

            /** @brief Returns two tensors of every data type, the first of type + 1 elements. */
            std::vector<Tensor> two_tensors_of_every_type()
            {
                std::vector<Tensor> sources;
                for (std::size_t type = 0; type < segment_rows().size(); ++type)
                {
                    const Core::DataType dtype = segment_rows()[type].dtype;
                    sources.push_back(shared_tensor(dtype, static_cast<std::int64_t>(type + 1)));
                    sources.push_back(shared_tensor(dtype, 2));
                }
                return sources;
            }
        } // namespace

        /** @brief Groups the tests that pack tensors on the program-wide runtime. */
        class Medium_TensorPack : public ::testing::Test
        {

        };

        /** @brief Tests that a gather then a scatter returns the bytes of three tensors of three types. */
        TEST_F(Medium_TensorPack, GatherThenScatterCopiesEveryByte)
        {
            std::vector<Tensor> sources = {shared_tensor(Core::DataType::Float64, 9),
                                           shared_tensor(Core::DataType::Half, 5),
                                           shared_tensor(Core::DataType::Int32, 3)};
            std::vector<Tensor> targets = {shared_tensor(Core::DataType::Float64, 9),
                                           shared_tensor(Core::DataType::Half, 5),
                                           shared_tensor(Core::DataType::Int32, 3)};
            fill_pattern(sources[0], 1);
            fill_pattern(sources[1], 8);
            fill_pattern(sources[2], 15);
            std::memset(targets[0].data(), 0, byte_count(targets[0]));
            std::memset(targets[1].data(), 0, byte_count(targets[1]));
            std::memset(targets[2].data(), 0, byte_count(targets[2]));
            const TensorPack pack(runtime(), layout_of(sources), 0);

            gather_then_scatter(pack, sources, targets);

            EXPECT_TRUE(same_bytes(sources[0], targets[0]));
            EXPECT_TRUE(same_bytes(sources[1], targets[1]));
            EXPECT_TRUE(same_bytes(sources[2], targets[2]));
        }

        /** @brief Tests that the tensors of a pack keep the gathered bytes after the pack is gone. */
        TEST_F(Medium_TensorPack, EntryViewKeepsTheGatheredBytesAfterThePackDies)
        {
            std::vector<Tensor> sources = {shared_tensor(Core::DataType::Float64, 9),
                                           shared_tensor(Core::DataType::Half, 5),
                                           shared_tensor(Core::DataType::Float64, 2),
                                           shared_tensor(Core::DataType::Int32, 3)};
            fill_pattern(sources[0], 3);
            fill_pattern(sources[1], 8);
            fill_pattern(sources[2], 13);
            fill_pattern(sources[3], 18);

            const std::vector<Tensor> views = entry_views_after_the_pack_dies(sources);

            ASSERT_EQ(views.size(), 4U);
            EXPECT_TRUE(same_bytes(views[0], sources[0]));
            EXPECT_TRUE(same_bytes(views[1], sources[1]));
            EXPECT_TRUE(same_bytes(views[2], sources[2]));
            EXPECT_TRUE(same_bytes(views[3], sources[3]));
        }

        /** @brief Tests that an entry view has the dtype, count and extent of its source. */
        TEST_F(Medium_TensorPack, EntryViewHasTheShapeOfItsSource)
        {
            const std::vector<Tensor> sources = {shared_tensor(Core::DataType::Half, 5),
                                                 shared_tensor(Core::DataType::Int32, 3)};
            const TensorPack pack(runtime(), layout_of(sources), 0);

            const Tensor second = pack.tensor(1);

            EXPECT_EQ(second.dtype, Core::DataType::Int32);
            EXPECT_EQ(second.num_elements, 3);
            EXPECT_EQ(second.shape[0], 3);
        }

        /** @brief Tests that tensor(index) past the entry count throws out_of_range. */
        TEST_F(Medium_TensorPack, TensorPastTheEntryCountThrows)
        {
            const std::vector<Tensor> sources = {shared_tensor(Core::DataType::Half, 5),
                                                 shared_tensor(Core::DataType::Int32, 3)};
            const TensorPack pack(runtime(), layout_of(sources), 0);

            EXPECT_THROW((void)pack.tensor(2), std::out_of_range);
        }

        /** @brief Runs one row of segment_rows() against a pack holding two tensors of every type. */
        class Medium_TensorPackSegment : public ::testing::TestWithParam<SegmentRow>
        {

        };

        TEST_P(Medium_TensorPackSegment, TensorHasTheSegmentsSizeAndOffset)
        {
            const TensorPack pack(runtime(), layout_of(two_tensors_of_every_type()), 0);
            const Core::DataType dtype = GetParam().dtype;
            const PackSegment& segment = pack.layout().segment(dtype);

            const Tensor view = pack.segment_tensor(dtype);

            EXPECT_EQ(view.rank, 1);
            EXPECT_EQ(view.dtype, dtype);
            EXPECT_EQ(static_cast<std::size_t>(view.num_elements), segment.element_count);
            EXPECT_EQ(byte_count(view), segment.size_bytes);
            EXPECT_EQ(static_cast<const std::byte*>(view.data()),
                      static_cast<const std::byte*>(pack.data()) + segment.offset_bytes);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_TensorPackSegment, ::testing::ValuesIn(segment_rows()),
            [](const ::testing::TestParamInfo<SegmentRow>& info)
            { return std::string(info.param.name); });

        /** @brief Groups the regression tests that need no second device. */
        class Medium_Regression_TensorPack : public ::testing::Test
        {

        };

        /** @brief Tests that a device index beyond the runtime's device count is refused. */
        TEST_F(Medium_Regression_TensorPack, PackRefusesADeviceIndexBeyondTheDeviceCount)
        {
            const std::vector<Tensor> shapes = {shared_tensor(Core::DataType::Float32, 4)};

            EXPECT_THROW(TensorPack(runtime(), layout_of(shapes),
                                    runtime().advanced().device_count()),
                         std::runtime_error);
        }

        /** @brief Runs the device test for device 0 and device 1, skipping a device the runtime lacks. */
        class Medium_Regression_TensorPackDevice : public ::testing::TestWithParam<std::size_t>
        {
            protected:
                void SetUp() override
                {
                    if (GetParam() >= runtime().advanced().device_count())
                        GTEST_SKIP() << "The runtime drives fewer devices than this row needs.";
                }
        };

        TEST_P(Medium_Regression_TensorPackDevice, BufferLivesOnTheDeviceItIsGiven)
        {
            const std::size_t device = GetParam();
            const std::vector<Tensor> shapes = {
                shared_tensor(Core::DataType::Float32, 4, device)};

            const TensorPack pack(runtime(), layout_of(shapes), device);

            EXPECT_EQ(pack.device_index(), device);
            EXPECT_EQ(pack.tensor(0).storage->device_index(), device);
            EXPECT_EQ(pack.tensor(0).storage->residency(), SushiRuntime::Residency::Host);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_Regression_TensorPackDevice,
            ::testing::Values(std::size_t{0}, std::size_t{1}),
            [](const ::testing::TestParamInfo<std::size_t>& info)
            { return "Device" + std::to_string(info.param); });
    } // namespace Test
} // namespace SushiBLAS
