/**************************************************************************/
/* test_shape_only_tensor.cpp                                             */
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
 * @file test_shape_only_tensor.cpp
 * @brief Tests that a shape-only Tensor carries dims, strides, dtype and layout and no storage.
 * @author Mustafa Garip
 */

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Returns dims as the span the builders take.
        static SushiRuntime::span<const int64_t> span_of(const std::vector<int64_t>& dims)
        {
            return SushiRuntime::span<const int64_t>(dims.data(), dims.size());
        }

        TEST(Small_ShapeOnlyTensor, ItCarriesTheShapeTheDtypeAndTheLayoutAsked)
        {
            const std::vector<int64_t> dims{2, 3, 4};

            const Tensor tensor = Tensor::make_shape_only(span_of(dims), Core::DataType::Float64,
                                                          Core::Layout::ColumnMajor);

            EXPECT_EQ(tensor.rank, 3);
            EXPECT_EQ(tensor.shape[0], int64_t{2});
            EXPECT_EQ(tensor.shape[1], int64_t{3});
            EXPECT_EQ(tensor.shape[2], int64_t{4});
            EXPECT_EQ(tensor.num_elements, int64_t{24});
            EXPECT_EQ(tensor.dtype, Core::DataType::Float64);
            EXPECT_EQ(tensor.layout, Core::Layout::ColumnMajor);
            EXPECT_TRUE(tensor.shape_only);
        }

        TEST(Small_ShapeOnlyTensor, ItsStridesAreThoseOfAnAllocatedTensorOfTheSameShape)
        {
            const std::vector<int64_t> dims{2, 3, 4};

            const Tensor tensor = Tensor::make_shape_only(span_of(dims));

            EXPECT_EQ(tensor.strides, Tensor::layout_strides(span_of(dims), Core::Layout::RowMajor));
            EXPECT_TRUE(tensor.is_contiguous());
        }

        TEST(Small_ShapeOnlyTensor, ItKeepsTheStridesItIsGiven)
        {
            const std::vector<int64_t> dims{3, 2};
            const std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{10, 2};

            const Tensor tensor = Tensor::make_shape_only(span_of(dims), strides,
                                                          Core::DataType::Int32,
                                                          Core::Layout::RowMajor);

            EXPECT_EQ(tensor.strides, strides);
            EXPECT_FALSE(tensor.is_contiguous());
        }

        TEST(Small_ShapeOnlyTensor, ItHasNoStorage)
        {
            const std::vector<int64_t> dims{2, 2};

            const Tensor tensor = Tensor::make_shape_only(span_of(dims));

            EXPECT_FALSE(static_cast<bool>(tensor.storage));
            EXPECT_EQ(tensor.data_or_null(), nullptr);
        }

        TEST(Small_ShapeOnlyTensor, ItsDataIsRefused)
        {
            const std::vector<int64_t> dims{2, 2};
            const Tensor tensor = Tensor::make_shape_only(span_of(dims));

            EXPECT_THROW((void)tensor.data(), std::runtime_error);
        }

        TEST(Small_ShapeOnlyTensor, ItsDeviceIsRefused)
        {
            const std::vector<int64_t> dims{2, 2};
            const Tensor tensor = Tensor::make_shape_only(span_of(dims));

            EXPECT_THROW((void)tensor.device(), std::runtime_error);
        }

        TEST(Small_ShapeOnlyTensor, ItsRegionIsEmpty)
        {
            const std::vector<int64_t> dims{2, 2};
            const Tensor tensor = Tensor::make_shape_only(span_of(dims));

            EXPECT_FALSE(tensor.region().valid());
        }

        TEST(Small_ShapeOnlyTensor, ATransposeStaysShapeOnlyWithSwappedExtents)
        {
            const std::vector<int64_t> dims{2, 5};
            const Tensor tensor = Tensor::make_shape_only(span_of(dims));

            const Tensor transposed = tensor.transpose(0, 1);

            EXPECT_TRUE(transposed.shape_only);
            EXPECT_EQ(transposed.shape[0], int64_t{5});
            EXPECT_EQ(transposed.shape[1], int64_t{2});
        }

        TEST(Small_ShapeOnlyTensor, AReshapeStaysShapeOnly)
        {
            const std::vector<int64_t> dims{2, 6};
            const Tensor tensor = Tensor::make_shape_only(span_of(dims));

            const Tensor reshaped = tensor.reshape({3, 4});

            EXPECT_TRUE(reshaped.shape_only);
            EXPECT_EQ(reshaped.shape[0], int64_t{3});
            EXPECT_EQ(reshaped.shape[1], int64_t{4});
        }

        TEST(Small_ShapeOnlyTensor, ASliceStaysShapeOnly)
        {
            const std::vector<int64_t> dims{6, 2};
            const Tensor tensor = Tensor::make_shape_only(span_of(dims));

            const Tensor sliced = tensor.slice(0, 1, 4);

            EXPECT_TRUE(sliced.shape_only);
            EXPECT_EQ(sliced.shape[0], int64_t{3});
        }

        TEST(Small_ShapeOnlyTensor, ANegativeDimensionIsRefused)
        {
            const std::vector<int64_t> dims{2, -1};

            EXPECT_THROW((void)Tensor::make_shape_only(span_of(dims)), std::runtime_error);
        }

        TEST(Small_ShapeOnlyTensor, ANegativeStrideIsRefused)
        {
            const std::vector<int64_t> dims{2};
            const std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{-1};

            EXPECT_THROW((void)Tensor::make_shape_only(span_of(dims), strides,
                                                       Core::DataType::Float32,
                                                       Core::Layout::RowMajor),
                         std::runtime_error);
        }

        TEST(Small_ShapeOnlyTensor, ARankPastTheLimitIsRefused)
        {
            const std::vector<int64_t> dims(Core::K_MAX_TENSOR_RANK + 1, 1);

            EXPECT_THROW((void)Tensor::make_shape_only(span_of(dims)), std::runtime_error);
        }

        TEST(Small_ShapeOnlyTensor, ADefaultTensorIsNotShapeOnly)
        {
            const Tensor tensor;

            EXPECT_FALSE(tensor.shape_only);
        }
    } // namespace Test
} // namespace SushiBLAS
