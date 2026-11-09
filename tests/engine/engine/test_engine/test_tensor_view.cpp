/**************************************************************************/
/* test_tensor_view.cpp                                                   */
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
 * @file test_tensor_view.cpp
 * @brief Tests that a tensor view's offset and strides lead to the elements it occupies.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the strides of a tensor, one per axis up to the highest rank.
            using Strides = std::array<int64_t, Core::K_MAX_TENSOR_RANK>;

            /** @brief Names one parent a slice is taken from, with the strides its layout fixes. */
            struct SliceRow
            {
                const char* name;             /**< The row name a failure shows. */
                Core::Layout layout;          /**< The layout of the parent. */
                std::vector<int64_t> dims;    /**< The shape of the parent. */
                std::vector<int64_t> strides; /**< The strides the layout gives that shape, by hand. */
                int32_t axis;                 /**< The axis the slice narrows. */
            };

            /** @brief Returns one row per axis of a row-major matrix, a column-major matrix and a rank-three tensor. */
            std::vector<SliceRow> slice_rows()
            {
                const Core::Layout row = Core::Layout::RowMajor;
                const Core::Layout column = Core::Layout::ColumnMajor;
                return {{"RowMajorAxis0", row, {4, 4}, {4, 1}, 0},
                        {"RowMajorAxis1", row, {4, 4}, {4, 1}, 1},
                        {"ColumnMajorAxis0", column, {4, 4}, {1, 4}, 0},
                        {"ColumnMajorAxis1", column, {4, 4}, {1, 4}, 1},
                        {"RankThreeAxis0", row, {3, 4, 5}, {20, 5, 1}, 0},
                        {"RankThreeAxis1", row, {3, 4, 5}, {20, 5, 1}, 1},
                        {"RankThreeAxis2", row, {3, 4, 5}, {20, 5, 1}, 2}};
            }

            /** @brief Returns the first rank entries of a tensor's strides. */
            std::vector<int64_t> strides_of(const Tensor& tensor)
            {
                return std::vector<int64_t>(tensor.strides.begin(),
                                            tensor.strides.begin() + tensor.rank);
            }

            /** @brief Writes element k = k into a contiguous float tensor's allocation. */
            void fill_with_indices(const Tensor& tensor)
            {
                float* buffer = static_cast<float*>(tensor.storage->allocation_base());
                std::iota(buffer, buffer + tensor.num_elements, 0.0f);
            }

            /**
             * @brief Expects every element of a slice to sit where its parent's strides put it.
             *
             * The allocation holds k at offset k, so the value at the offset the view reports
             * is the offset the parent's literal strides give that element.
             */
            void expect_view_addresses_its_elements(const Tensor& view, const Tensor& parent,
                                                    const std::vector<int64_t>& parent_strides,
                                                    int32_t axis, int64_t start)
            {
                const float* buffer = static_cast<const float*>(parent.storage->allocation_base());
                std::array<int64_t, Core::K_MAX_TENSOR_RANK> coordinate{};

                for (int64_t flat = 0; flat < view.num_elements; ++flat)
                {
                    int64_t rest = flat;
                    for (int32_t d = view.rank - 1; d >= 0; --d)
                    {
                        coordinate[static_cast<std::size_t>(d)] = rest % view.shape[d];
                        rest /= view.shape[d];
                    }

                    int64_t reported = view.storage_offset;
                    int64_t truth = 0;
                    for (int32_t d = 0; d < view.rank; ++d)
                    {
                        const int64_t index = coordinate[static_cast<std::size_t>(d)];
                        reported += index * view.strides[d];
                        truth += (index + (d == axis ? start : 0)) *
                                 parent_strides[static_cast<std::size_t>(d)];
                    }

                    ASSERT_GE(reported, 0);
                    ASSERT_LT(reported, parent.num_elements) << "element " << flat;
                    EXPECT_EQ(buffer[reported], static_cast<float>(truth)) << "element " << flat;
                }
            }

            /** @brief Names one view whose byte region is stated by hand. */
            struct RegionRow
            {
                const char* name;          /**< The row name a failure shows. */
                std::vector<int64_t> dims; /**< The shape of the parent. */
                int32_t axis;              /**< The axis narrowed to {1, 3}, or -1 for no slice. */
                int64_t elements;          /**< The element count of the view. */
                int64_t span;              /**< The elements from the first reached to past the last. */
                std::size_t offset_bytes;  /**< Where the region starts in the allocation. */
                std::size_t length_bytes;  /**< How long the region is. */
            };

            /** @brief Returns the whole-tensor row and the inner-axis slice row. */
            std::vector<RegionRow> region_rows()
            {
                // The inner slice holds offsets 1,2, 5,6, 9,10, 13,14: 8 elements over 14.
                return {{"ContiguousTensor", {4, 5}, -1, 20, 20, 0, 80},
                        {"InnerAxisSlice", {4, 4}, 1, 8, 14, 4, 56}};
            }
        } // namespace

        /** @brief Runs one row of slice_rows() on a host engine of the row's layout. */
        class Small_TensorViewSlice : public HostEngineTest,
                                      public ::testing::WithParamInterface<SliceRow>
        {
            protected:
                /** @brief Rebuilds the engine in the row's layout. */
                void SetUp() override
                {
                    HostEngineTest::SetUp();
                    reinit_engine(GetParam().layout);
                }
        };

        TEST_P(Small_TensorViewSlice, AddressesTheElementsItsParentPlaced)
        {
            const SliceRow row = GetParam();
            const Tensor parent = engine->create_tensor(row.dims);
            ASSERT_EQ(strides_of(parent), row.strides);
            fill_with_indices(parent);

            const Tensor view = parent.slice(row.axis, 1, 3);

            expect_view_addresses_its_elements(view, parent, row.strides, row.axis, 1);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_TensorViewSlice, ::testing::ValuesIn(slice_rows()),
            [](const ::testing::TestParamInfo<SliceRow>& info)
            { return std::string(info.param.name); });

        /** @brief Groups the single-case view tests on a host engine. */
        class Small_TensorView : public HostEngineTest
        {

        };

        /** @brief Tests that reshape refuses the non-contiguous view an inner-axis slice gives. */
        TEST_F(Small_TensorView, ReshapeRefusesAnInnerAxisSlice)
        {
            const Tensor parent = engine->create_tensor({4, 4});

            const Tensor inner = parent.slice(1, 1, 3);

            EXPECT_THROW((void)inner.reshape({8}), std::runtime_error);
        }

        /** @brief Tests that a transposed 3 by 5 matrix spans its 15 elements. */
        TEST_F(Small_TensorView, TransposeSpansItsWholeMatrix)
        {
            const Tensor parent = engine->create_tensor({3, 5});

            const Tensor transposed = parent.transpose(0, 1);

            EXPECT_EQ(transposed.shape[0], 5);
            EXPECT_EQ(transposed.shape[1], 3);
            EXPECT_EQ(transposed.element_span(), 15);
        }

        /** @brief Tests that the stride-explicit constructor keeps strides inside the capacity. */
        TEST_F(Small_TensorView, StrideExplicitConstructorKeepsCoherentStrides)
        {
            const Tensor parent = engine->create_tensor({4, 4});
            const std::vector<int64_t> dims = {4, 4};
            const Strides contiguous = {4, 1};

            const Tensor view(parent.storage, dims, contiguous, Core::DataType::Float32, 0,
                              Core::Layout::RowMajor);

            EXPECT_EQ(strides_of(view), (std::vector<int64_t>{4, 1}));
            EXPECT_EQ(view.element_span(), 16);
        }

        /** @brief Tests that the stride-explicit constructor refuses a negative stride. */
        TEST_F(Small_TensorView, StrideExplicitConstructorRefusesANegativeStride)
        {
            const Tensor parent = engine->create_tensor({4, 4});
            const std::vector<int64_t> dims = {4, 4};
            const Strides negative = {4, -1};

            EXPECT_THROW((void)Tensor(parent.storage, dims, negative, Core::DataType::Float32, 0,
                                      Core::Layout::RowMajor),
                         std::runtime_error);
        }

        /** @brief Tests that the stride-explicit constructor refuses a span past the capacity. */
        TEST_F(Small_TensorView, StrideExplicitConstructorRefusesASpanPastTheCapacity)
        {
            const Tensor parent = engine->create_tensor({4, 4});
            const std::vector<int64_t> dims = {4, 4};
            // Rows 64 apart reach element 3 * 64 + 3 of a 16-element allocation.
            const Strides too_wide = {64, 1};

            EXPECT_THROW((void)Tensor(parent.storage, dims, too_wide, Core::DataType::Float32, 0,
                                      Core::Layout::RowMajor),
                         std::runtime_error);
        }

        /** @brief Runs one axis of a whole-axis slice of a 4 by 5 matrix. */
        class Small_TensorViewIdentity : public HostEngineTest,
                                         public ::testing::WithParamInterface<int32_t>
        {

        };

        TEST_P(Small_TensorViewIdentity, WholeAxisSliceIsTheParent)
        {
            const Tensor parent = engine->create_tensor({4, 5});

            const Tensor view = parent.slice(GetParam(), 0, parent.shape[GetParam()]);

            EXPECT_EQ(parent.shape, view.shape);
            EXPECT_EQ(parent.strides, view.strides);
            EXPECT_EQ(parent.storage_offset, view.storage_offset);
            EXPECT_TRUE(view.is_contiguous());
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_TensorViewIdentity, ::testing::Values(0, 1),
            [](const ::testing::TestParamInfo<int32_t>& info)
            { return "Axis" + std::to_string(info.param); });

        /** @brief Runs one row of region_rows() on a host engine. */
        class Small_TensorViewRegion : public HostEngineTest,
                                       public ::testing::WithParamInterface<RegionRow>
        {

        };

        TEST_P(Small_TensorViewRegion, CoversEveryByteTheViewReaches)
        {
            const RegionRow row = GetParam();
            const Tensor parent = engine->create_tensor(row.dims);
            const Tensor view = row.axis < 0 ? parent : parent.slice(row.axis, 1, 3);

            const auto region = view.region();

            EXPECT_EQ(view.num_elements, row.elements);
            EXPECT_EQ(view.element_span(), row.span);
            EXPECT_EQ(region.base, parent.storage->allocation_base());
            EXPECT_EQ(region.offset, row.offset_bytes);
            EXPECT_EQ(region.length, row.length_bytes);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_TensorViewRegion, ::testing::ValuesIn(region_rows()),
            [](const ::testing::TestParamInfo<RegionRow>& info)
            { return std::string(info.param.name); });
    } // namespace Test
} // namespace SushiBLAS
