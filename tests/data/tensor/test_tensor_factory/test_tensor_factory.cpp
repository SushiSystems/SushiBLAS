/**************************************************************************/
/* test_tensor_factory.cpp                                                */
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
 * @file test_tensor_factory.cpp
 * @brief Tests the layout, the byte size and the tensors a TensorFactory creates.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Returns dims as the span the factory takes.
        static SushiRuntime::span<const int64_t> span_of(const std::vector<int64_t>& dims)
        {
            return SushiRuntime::span<const int64_t>(dims.data(), dims.size());
        }

        /// @brief A shape, an element type and the byte count a dense tensor of them needs.
        struct AllocationRow
        {
            std::vector<int64_t> dims;
            Core::DataType dtype;
            std::size_t bytes;
        };

        class Small_TensorFactoryAllocationBytes : public ::testing::TestWithParam<AllocationRow>
        {
        };

        TEST_P(Small_TensorFactoryAllocationBytes, IsTheElementCountTimesTheElementSize)
        {
            const AllocationRow row = GetParam();

            EXPECT_EQ(TensorFactory::allocation_bytes(span_of(row.dims), row.dtype), row.bytes);
        }

        // 3 * 5 doubles of 8 bytes; a zero extent gives no bytes.
        INSTANTIATE_TEST_SUITE_P(
            , Small_TensorFactoryAllocationBytes,
            ::testing::Values(AllocationRow{{3, 5}, Core::DataType::Float64, 120},
                              AllocationRow{{4, 0}, Core::DataType::Float32, 0}));

        TEST(Small_TensorFactory, AllocationBytesRefusesANegativeExtent)
        {
            const std::vector<int64_t> dims{-1};

            EXPECT_THROW((void)TensorFactory::allocation_bytes(span_of(dims),
                                                                Core::DataType::Float32),
                         std::runtime_error);
        }

        TEST(Small_TensorFactory, AllocationBytesRefusesAnOverflowingCount)
        {
            constexpr int64_t K_HUGE = int64_t{1} << 40;
            const std::vector<int64_t> square{K_HUGE, K_HUGE};
            const std::vector<int64_t> wide{int64_t{1} << 62};

            EXPECT_THROW((void)TensorFactory::allocation_bytes(span_of(square),
                                                                Core::DataType::Float32),
                         std::runtime_error);
            EXPECT_THROW((void)TensorFactory::allocation_bytes(span_of(wide),
                                                                Core::DataType::Float64),
                         std::runtime_error);
        }

        TEST(Small_TensorFactory, ACreatedTensorCountsItsElements)
        {
            const TensorFactory factory(TensorFactory::Host{});
            const std::vector<int64_t> dims{2, 3};

            const Tensor tensor = factory.create(span_of(dims), Core::DataType::Complex32);

            EXPECT_EQ(tensor.num_elements, int64_t{6});
        }

        TEST(Small_TensorFactory, ACreatedTensorsStridesFollowTheFactorysLayout)
        {
            const TensorFactory factory(TensorFactory::Host{}, Core::Layout::ColumnMajor);
            const std::vector<int64_t> dims{2, 3};

            const Tensor tensor = factory.create(span_of(dims));

            EXPECT_EQ(tensor.strides[0], int64_t{1});
            EXPECT_EQ(tensor.strides[1], int64_t{2});
        }

        TEST(Small_TensorFactory, ATensorWithoutElementsKeepsTheShapeAsked)
        {
            const TensorFactory factory(TensorFactory::Host{});
            const std::vector<int64_t> dims{0, 7};

            const Tensor tensor = factory.create(span_of(dims));

            EXPECT_EQ(tensor.shape[0], int64_t{0});
            EXPECT_EQ(tensor.shape[1], int64_t{7});
        }

        TEST(Medium_TensorFactory, PlacesTheStorageOnTheGivenDevice)
        {
            const TensorFactory factory(TestEnvironment::runtime());
            const std::size_t last = TestEnvironment::device_count() - 1;
            const std::vector<int64_t> dims{4};

            const Tensor tensor = factory.create(span_of(dims), Core::DataType::Float32,
                                                 SushiRuntime::Residency::Shared, last);

            EXPECT_EQ(tensor.storage->device_index(), last);
        }

        TEST(Medium_TensorFactory, PlacesTheStorageInTheGivenResidency)
        {
            const TensorFactory factory(TestEnvironment::runtime());
            const std::vector<int64_t> dims{4};

            const Tensor tensor = factory.create(span_of(dims), Core::DataType::Float32,
                                                 SushiRuntime::Residency::Device);

            EXPECT_EQ(tensor.storage->residency(), SushiRuntime::Residency::Device);
        }

        TEST(Medium_TensorFactoryPlanned, APlannedTensorHasAHandleAndNoHostAddress)
        {
            const TensorFactory factory(TestEnvironment::runtime());
            const std::vector<int64_t> dims{4, 8};

            const Tensor tensor = factory.planned(span_of(dims), Core::DataType::Float32);

            EXPECT_FALSE(tensor.global_handle().is_null());
            EXPECT_EQ(tensor.storage->residency(), SushiRuntime::Residency::Planned);
            EXPECT_EQ(tensor.storage->allocation_base(), nullptr);
            EXPECT_THROW((void)tensor.data(), std::runtime_error);
        }

        TEST(Medium_TensorFactoryPlanned, APlannedTensorReadsBackWhatTheHostWrote)
        {
            const TensorFactory factory(TestEnvironment::runtime());
            const std::vector<int64_t> dims{2, 3};
            const Tensor tensor = factory.planned(span_of(dims), Core::DataType::Float32);
            const std::vector<float> written{1.0f, -2.0f, 3.5f, 4.0f, 0.25f, -6.0f};

            tensor.storage->copy_from_host(written.data(), 0, written.size() * sizeof(float));
            std::vector<float> read(written.size());
            tensor.storage->copy_to_host(read.data(), 0, read.size() * sizeof(float));

            EXPECT_EQ(read, written);
        }

        TEST(Medium_TensorFactoryPlanned, RefusesAShapeWithNoElement)
        {
            const TensorFactory factory(TestEnvironment::runtime());
            const std::vector<int64_t> dims{4, 0};

            EXPECT_THROW((void)factory.planned(span_of(dims)), std::runtime_error);
        }

        TEST(Small_TensorFactoryPlanned, AHostFactoryRefusesAPlannedTensor)
        {
            const TensorFactory factory(TensorFactory::Host{});
            const std::vector<int64_t> dims{4};

            EXPECT_THROW((void)factory.planned(span_of(dims)), std::runtime_error);
            EXPECT_THROW((void)factory.from_file("unused.bin", 0, span_of(dims)),
                         std::runtime_error);
        }

        TEST(Medium_TensorFactoryPlanned, AFileTensorReadsItsFileRange)
        {
            const std::vector<float> contents{9.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 7.0f};
            const std::string path = ::testing::TempDir() + "sushiblas_tensor_factory_range.bin";
            {
                std::ofstream out(path, std::ios::binary | std::ios::trunc);
                out.write(reinterpret_cast<const char*>(contents.data()),
                          static_cast<std::streamsize>(contents.size() * sizeof(float)));
            }
            const TensorFactory factory(TestEnvironment::runtime());
            const std::vector<int64_t> dims{2, 2};

            const Tensor tensor = factory.from_file(path, 2 * sizeof(float), span_of(dims));
            std::vector<float> read(4);
            tensor.storage->copy_to_host(read.data(), 0, read.size() * sizeof(float));
            std::remove(path.c_str());

            EXPECT_FALSE(tensor.global_handle().is_null());
            EXPECT_EQ(read, (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
        }
    } // namespace Test
} // namespace SushiBLAS
