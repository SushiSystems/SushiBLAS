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
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Returns a storage of n_bytes on the shared runtime's first device.
        static SushiRuntime::sushi_ptr<Storage> bytes(std::size_t n_bytes)
        {
            return SushiRuntime::make_sushi<Storage>(TestEnvironment::runtime(), n_bytes);
        }

        /// @brief Returns the strides of dims in layout, rank entries long.
        static std::vector<int64_t> strides_of(std::vector<int64_t> dims, Core::Layout layout)
        {
            const auto all = Tensor::layout_strides(
                SushiRuntime::span<const int64_t>(dims.data(), dims.size()), layout);
            return std::vector<int64_t>(all.begin(), all.begin() + dims.size());
        }

        TEST(Unit_Tensor, RowMajorStridesRunFromTheLastAxis)
        {
            EXPECT_EQ(strides_of({2, 3, 4}, Core::Layout::RowMajor),
                      (std::vector<int64_t>{12, 4, 1}));
        }

        TEST(Unit_Tensor, ColumnMajorTransposesTheMatrixAxesOnly)
        {
            EXPECT_EQ(strides_of({3, 4}, Core::Layout::ColumnMajor), (std::vector<int64_t>{1, 3}));
            EXPECT_EQ(strides_of({2, 3, 4}, Core::Layout::ColumnMajor),
                      (std::vector<int64_t>{12, 1, 3}));
            EXPECT_EQ(strides_of({5}, Core::Layout::ColumnMajor), (std::vector<int64_t>{1}));
        }

        TEST(Unit_Tensor, StridesRefuseANegativeDimensionAndTooManyAxes)
        {
            EXPECT_THROW((void)strides_of({2, -1}, Core::Layout::RowMajor), std::runtime_error);
            EXPECT_THROW((void)strides_of({1, 1, 1, 1, 1, 1, 1}, Core::Layout::RowMajor),
                         std::runtime_error);
        }

        TEST(Unit_Tensor, CountsItsElementsAndChecksItsCapacity)
        {
            const Tensor fits(bytes(24), {2, 3}, Core::DataType::Float32);

            EXPECT_EQ(fits.rank, 2);
            EXPECT_EQ(fits.num_elements, int64_t{6});
            EXPECT_TRUE(fits.is_contiguous());
            EXPECT_TRUE(fits.is_aligned());
            EXPECT_THROW(Tensor(bytes(128), {2, 3}, Core::DataType::Float64, 11),
                         std::runtime_error);
        }

        TEST(Unit_Tensor, ATransposeSwapsShapeAndStridesAndIsNotContiguous)
        {
            const Tensor matrix(bytes(24), {2, 3});
            const Tensor transposed = matrix.transpose(0, 1);

            EXPECT_EQ(transposed.shape[0], int64_t{3});
            EXPECT_EQ(transposed.strides[0], int64_t{1});
            EXPECT_EQ(transposed.strides[1], int64_t{3});
            EXPECT_FALSE(transposed.is_contiguous());
            EXPECT_EQ(transposed.data(), matrix.data());
            EXPECT_THROW((void)matrix.transpose(0, 2), std::runtime_error);
        }

        TEST(Unit_Tensor, ASliceMovesTheOffsetAndKeepsTheStrides)
        {
            const Tensor matrix(bytes(48), {3, 4});
            const Tensor rows = matrix.slice(0, 1, 3);
            const Tensor columns = matrix.slice(1, -2, 4);

            EXPECT_EQ(rows.storage_offset, int64_t{4});
            EXPECT_EQ(rows.shape[0], int64_t{2});
            EXPECT_EQ(columns.storage_offset, int64_t{2});
            EXPECT_EQ(columns.shape[1], int64_t{2});
            EXPECT_EQ(columns.strides[0], int64_t{4});
            EXPECT_FALSE(columns.is_contiguous());
            EXPECT_EQ(rows.data_as<float>(), matrix.data_as<float>() + 4);
        }

        TEST(Unit_Tensor, AReshapeKeepsTheElementsOfAContiguousTensor)
        {
            const Tensor matrix(bytes(48), {3, 4});
            const Tensor flat = matrix.reshape({12});
            const Tensor cube = matrix.reshape(std::vector<int64_t>{2, 2, 3});

            EXPECT_EQ(flat.rank, 1);
            EXPECT_EQ(flat.num_elements, int64_t{12});
            EXPECT_EQ(cube.rank, 3);
            EXPECT_EQ(cube.data(), matrix.data());
            EXPECT_THROW((void)matrix.reshape({5}), std::runtime_error);
        }

        TEST(Unit_Tensor, TheElementSpanOfAViewReachesItsLastElement)
        {
            const Tensor matrix(bytes(48), {3, 4});
            const Tensor column = matrix.slice(1, 1, 2);

            EXPECT_EQ(matrix.element_span(), int64_t{12});
            EXPECT_EQ(column.num_elements, int64_t{3});
            EXPECT_EQ(column.element_span(), int64_t{9});
        }

        TEST(Unit_Tensor, ATensorWithoutStorageHasNoData)
        {
            const Tensor empty;

            EXPECT_EQ(empty.data_or_null(), nullptr);
            EXPECT_EQ(empty.num_elements, int64_t{0});
            EXPECT_THROW((void)empty.device(), std::runtime_error);
        }
    } // namespace Test
} // namespace SushiBLAS
