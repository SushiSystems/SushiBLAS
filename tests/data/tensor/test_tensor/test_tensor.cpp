/**************************************************************************/
/* test_tensor.cpp                                                        */
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
 * @file test_tensor.cpp
 * @brief Tests the strides, views, checks and regions of a Tensor over one storage.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/tensor/runtime_block.hpp>
#include <SushiBLAS/tensor/host_block.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include "support/owned_memory.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Returns the strides of dims in layout, rank entries long.
        static std::vector<int64_t> strides_of(std::vector<int64_t> dims, Core::Layout layout)
        {
            const auto all = Tensor::layout_strides(
                SushiRuntime::span<const int64_t>(dims.data(), dims.size()), layout);
            return std::vector<int64_t>(all.begin(), all.begin() + dims.size());
        }

        /// @brief A shape, a layout and the strides they give.
        struct StridesRow
        {
            std::vector<int64_t> dims;
            Core::Layout layout;
            std::vector<int64_t> strides;
        };

        class Small_TensorStrides : public ::testing::TestWithParam<StridesRow>
        {
        };

        TEST_P(Small_TensorStrides, FollowTheShapeUnderTheLayout)
        {
            const StridesRow row = GetParam();

            EXPECT_EQ(strides_of(row.dims, row.layout), row.strides);
        }

        // Row major runs from the last axis; column major swaps the two matrix axes.
        INSTANTIATE_TEST_SUITE_P(
            , Small_TensorStrides,
            ::testing::Values(
                StridesRow{{2, 3, 4}, Core::Layout::RowMajor, {12, 4, 1}},
                StridesRow{{3, 4}, Core::Layout::ColumnMajor, {1, 3}},
                StridesRow{{2, 3, 4}, Core::Layout::ColumnMajor, {12, 1, 3}},
                StridesRow{{5}, Core::Layout::ColumnMajor, {1}}));

        TEST(Small_Tensor, StridesRefuseANegativeDimension)
        {
            EXPECT_THROW((void)strides_of({2, -1}, Core::Layout::RowMajor), std::runtime_error);
        }

        TEST(Small_Tensor, StridesRefuseMoreAxesThanTheLimit)
        {
            EXPECT_THROW((void)strides_of(std::vector<int64_t>(Core::K_MAX_TENSOR_RANK + 1, 1),
                                          Core::Layout::RowMajor),
                         std::runtime_error);
        }

        /// @brief A tensor over a storage of one alignment unit: its element count and its offset.
        struct CapacityRow
        {
            int64_t elements;
            int64_t offset;
            bool accepted;
        };

        /// @brief Returns the Float32 element count that fills one alignment unit.
        static int64_t unit_elements()
        {
            return static_cast<int64_t>(K_STORAGE_ALIGNMENT / sizeof(float));
        }

        /// @brief Returns the rows of a tensor over a storage of one alignment unit.
        static std::vector<CapacityRow> capacity_rows()
        {
            return {{unit_elements(), 0, true},
                    {unit_elements() + 1, 0, false},
                    {unit_elements() - 1, 1, true},
                    {unit_elements(), 1, false}};
        }

        /// @brief Returns whether a Float32 tensor of the row fits the storage.
        static bool fits(const SushiRuntime::sushi_ptr<Storage>& storage, const CapacityRow& row)
        {
            try
            {
                const Tensor tensor(storage, {row.elements}, Core::DataType::Float32, row.offset);
                return true;
            }
            catch (const std::runtime_error&)
            {
                return false;
            }
        }

        class Small_TensorCapacity : public ::testing::TestWithParam<CapacityRow>
        {
        };

        TEST_P(Small_TensorCapacity, IsHeldToTheSizeOfAHostStorage)
        {
            const CapacityRow row = GetParam();

            EXPECT_EQ(fits(make_host_storage(K_STORAGE_ALIGNMENT), row), row.accepted);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_TensorCapacity, ::testing::ValuesIn(capacity_rows()));

        class Medium_TensorCapacityOverABorrowedStorage : public ::testing::TestWithParam<CapacityRow>
        {
        };

        TEST_P(Medium_TensorCapacityOverABorrowedStorage, IsHeldToTheSizeOfTheBuffer)
        {
            const OwnedMemory memory(K_STORAGE_ALIGNMENT);
            const auto storage = borrow_runtime_storage(
                TestEnvironment::runtime(), memory.borrow(0, K_STORAGE_ALIGNMENT));
            const CapacityRow row = GetParam();

            EXPECT_EQ(fits(storage, row), row.accepted);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_TensorCapacityOverABorrowedStorage,
                                 ::testing::ValuesIn(capacity_rows()));

        TEST(Small_Tensor, CountsItsRankAndElements)
        {
            const Tensor tensor(make_host_storage(24), {2, 3}, Core::DataType::Float32);

            EXPECT_EQ(tensor.rank, 2);
            EXPECT_EQ(tensor.num_elements, int64_t{6});
        }

        TEST(Small_Tensor, AFreshTensorIsContiguous)
        {
            const Tensor tensor(make_host_storage(24), {2, 3}, Core::DataType::Float32);

            EXPECT_TRUE(tensor.is_contiguous());
        }

        TEST(Small_Tensor, AFreshTensorOverAHostStorageIsAligned)
        {
            const Tensor tensor(make_host_storage(24), {2, 3}, Core::DataType::Float32);

            EXPECT_TRUE(tensor.is_aligned());
        }

        TEST(Small_Tensor, ATransposeSwapsTheShape)
        {
            const Tensor matrix(make_host_storage(24), {2, 3});

            const Tensor transposed = matrix.transpose(0, 1);

            EXPECT_EQ(transposed.shape[0], int64_t{3});
            EXPECT_EQ(transposed.shape[1], int64_t{2});
        }

        TEST(Small_Tensor, ATransposeSwapsTheStrides)
        {
            const Tensor matrix(make_host_storage(24), {2, 3});

            const Tensor transposed = matrix.transpose(0, 1);

            EXPECT_EQ(transposed.strides[0], int64_t{1});
            EXPECT_EQ(transposed.strides[1], int64_t{3});
        }

        TEST(Small_Tensor, ATransposeIsNotContiguous)
        {
            const Tensor matrix(make_host_storage(24), {2, 3});

            EXPECT_FALSE(matrix.transpose(0, 1).is_contiguous());
        }

        TEST(Small_Tensor, ATransposeViewsTheSameData)
        {
            const Tensor matrix(make_host_storage(24), {2, 3});

            EXPECT_EQ(matrix.transpose(0, 1).data(), matrix.data());
        }

        TEST(Small_Tensor, ATransposeRefusesAnAxisPastTheRank)
        {
            const Tensor matrix(make_host_storage(24), {2, 3});

            EXPECT_THROW((void)matrix.transpose(0, 2), std::runtime_error);
        }

        TEST(Small_Tensor, ASliceOfRowsMovesTheOffsetByWholeRows)
        {
            const Tensor matrix(make_host_storage(48), {3, 4});

            const Tensor rows = matrix.slice(0, 1, 3);

            EXPECT_EQ(rows.storage_offset, int64_t{4});
        }

        TEST(Small_Tensor, ASliceCutsTheAxisToTheRange)
        {
            const Tensor matrix(make_host_storage(48), {3, 4});

            const Tensor rows = matrix.slice(0, 1, 3);

            EXPECT_EQ(rows.shape[0], int64_t{2});
        }

        TEST(Small_Tensor, ASliceWithANegativeStartCountsFromTheEnd)
        {
            const Tensor matrix(make_host_storage(48), {3, 4});

            const Tensor columns = matrix.slice(1, -2, 4);

            EXPECT_EQ(columns.storage_offset, int64_t{2});
            EXPECT_EQ(columns.shape[1], int64_t{2});
        }

        TEST(Small_Tensor, ASliceKeepsTheStrides)
        {
            const Tensor matrix(make_host_storage(48), {3, 4});

            const Tensor columns = matrix.slice(1, 2, 4);

            EXPECT_EQ(columns.strides[0], int64_t{4});
            EXPECT_EQ(columns.strides[1], int64_t{1});
        }

        TEST(Small_Tensor, ASliceOfColumnsIsNotContiguous)
        {
            const Tensor matrix(make_host_storage(48), {3, 4});

            EXPECT_FALSE(matrix.slice(1, 2, 4).is_contiguous());
        }

        TEST(Small_Tensor, ASlicesDataIsTheBaseMovedByTheOffset)
        {
            const Tensor matrix(make_host_storage(48), {3, 4});

            const Tensor rows = matrix.slice(0, 1, 3);

            EXPECT_EQ(rows.data_as<float>(), matrix.data_as<float>() + 4);
        }

        TEST(Small_Tensor, AReshapeGivesTheNewRankAndElementCount)
        {
            const Tensor matrix(make_host_storage(48), {3, 4});

            const Tensor cube = matrix.reshape(std::vector<int64_t>{2, 2, 3});

            EXPECT_EQ(cube.rank, 3);
            EXPECT_EQ(cube.num_elements, int64_t{12});
        }

        TEST(Small_Tensor, AReshapeViewsTheSameData)
        {
            const Tensor matrix(make_host_storage(48), {3, 4});

            EXPECT_EQ(matrix.reshape({12}).data(), matrix.data());
        }

        TEST(Small_Tensor, AReshapeRefusesAChangeOfElementCount)
        {
            const Tensor matrix(make_host_storage(48), {3, 4});

            EXPECT_THROW((void)matrix.reshape({5}), std::runtime_error);
        }

        TEST(Small_Tensor, TheElementSpanOfAContiguousTensorIsItsElementCount)
        {
            const Tensor matrix = Tensor::make_shape_only(std::vector<int64_t>{3, 4});

            EXPECT_EQ(matrix.element_span(), int64_t{12});
        }

        TEST(Small_Tensor, TheElementSpanOfAViewReachesItsLastElement)
        {
            const Tensor matrix = Tensor::make_shape_only(std::vector<int64_t>{3, 4});

            const Tensor column = matrix.slice(1, 1, 2);

            EXPECT_EQ(column.element_span(), int64_t{9});
        }

        TEST(Small_Tensor, ATensorWithoutStorageHasNoDataPointer)
        {
            const Tensor empty;

            EXPECT_EQ(empty.data_or_null(), nullptr);
        }

        TEST(Small_Tensor, ATensorWithoutStorageHasNoElements)
        {
            const Tensor empty;

            EXPECT_EQ(empty.num_elements, int64_t{0});
        }

        TEST(Small_Tensor, ATensorWithoutStorageRefusesItsDevice)
        {
            const Tensor empty;

            EXPECT_THROW((void)empty.device(), std::runtime_error);
        }

        /// @brief Returns the two disjoint views and the overlapping one over a 32-element host storage.
        struct RegionViews
        {
            Tensor first;       ///< Elements 0 to 7.
            Tensor second;      ///< Elements 8 to 15.
            Tensor overlapping; ///< Elements 4 to 11.
        };

        /// @brief Returns views of 8 elements at the offsets 0, 8 and 4 of one 32-element storage.
        static RegionViews region_views()
        {
            auto storage = make_host_storage(32 * sizeof(float));

            return {Tensor(storage, {8}, Core::DataType::Float32, 0),
                    Tensor(storage, {8}, Core::DataType::Float32, 8),
                    Tensor(storage, {8}, Core::DataType::Float32, 4)};
        }

        TEST(Small_Tensor, TheRegionsOfDisjointViewsDoNotOverlap)
        {
            const RegionViews views = region_views();

            EXPECT_FALSE(views.first.region().overlaps(views.second.region()));
        }

        TEST(Small_Tensor, TheRegionsOfViewsThatShareElementsOverlap)
        {
            const RegionViews views = region_views();

            EXPECT_TRUE(views.first.region().overlaps(views.overlapping.region()));
            EXPECT_TRUE(views.second.region().overlaps(views.overlapping.region()));
        }

        TEST(Small_Tensor, ARegionOverlapsItself)
        {
            const RegionViews views = region_views();

            EXPECT_TRUE(views.first.region().overlaps(views.first.region()));
        }

        TEST(Small_Tensor, ASliceOffTheBaseReportsItsOwnAlignment)
        {
            constexpr int64_t K_ELEMENTS = 64;
            const Tensor whole(make_host_storage(K_ELEMENTS * sizeof(float)), {K_ELEMENTS},
                               Core::DataType::Float32);

            // One float past the base is 4 bytes in; 32 floats past it are 128 bytes in.
            EXPECT_FALSE(whole.slice(0, 1, K_ELEMENTS).is_aligned());
            EXPECT_TRUE(whole.slice(0, 32, K_ELEMENTS).is_aligned());
        }
    } // namespace Test
} // namespace SushiBLAS
