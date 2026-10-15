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
 * @file test_tensor_factory.cpp
 * @brief Tests the layout, the byte size and the tensors a TensorFactory creates.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
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
        SushiRuntime::span<const int64_t> span_of(const std::vector<int64_t>& dims)
        {
            return SushiRuntime::span<const int64_t>(dims.data(), dims.size());
        }

        TEST(Unit_TensorFactory, KeepsItsRuntimeAndLayout)
        {
            const TensorFactory row_major(TestEnvironment::runtime());
            const TensorFactory column_major(TestEnvironment::runtime(),
                                             Core::Layout::ColumnMajor);

            EXPECT_EQ(row_major.layout(), Core::Layout::RowMajor);
            EXPECT_EQ(column_major.layout(), Core::Layout::ColumnMajor);
            EXPECT_EQ(&row_major.runtime(), &TestEnvironment::runtime());
        }

        TEST(Unit_TensorFactory, AllocationBytesIsTheElementCountTimesTheElementSize)
        {
            const std::vector<int64_t> dims{3, 5};

            EXPECT_EQ(TensorFactory::allocation_bytes(span_of(dims), Core::DataType::Float64),
                      std::size_t{120});
            EXPECT_EQ(TensorFactory::allocation_bytes(span_of({4, 0}), Core::DataType::Float32),
                      std::size_t{0});
            EXPECT_THROW((void)TensorFactory::allocation_bytes(span_of({-1}),
                                                                Core::DataType::Float32),
                         std::runtime_error);
        }

        TEST(Unit_TensorFactory, CreatesATensorOfItsLayoutAndTheGivenType)
        {
            const TensorFactory factory(TestEnvironment::runtime(), Core::Layout::ColumnMajor);
            const std::vector<int64_t> dims{2, 3};
            const Tensor tensor = factory.create(span_of(dims), Core::DataType::Complex32);

            EXPECT_EQ(tensor.layout, Core::Layout::ColumnMajor);
            EXPECT_EQ(tensor.dtype, Core::DataType::Complex32);
            EXPECT_EQ(tensor.num_elements, int64_t{6});
            EXPECT_EQ(tensor.strides[0], int64_t{1});
            EXPECT_EQ(tensor.strides[1], int64_t{2});
            ASSERT_TRUE(tensor.storage);
            EXPECT_GE(tensor.storage->size_bytes(), std::size_t{48});
        }

        TEST(Unit_TensorFactory, ATensorWithoutElementsHasNoStorage)
        {
            const TensorFactory factory(TestEnvironment::runtime());
            const std::vector<int64_t> dims{0, 7};
            const Tensor tensor = factory.create(span_of(dims));

            EXPECT_FALSE(tensor.storage);
            EXPECT_EQ(tensor.num_elements, int64_t{0});
            EXPECT_EQ(tensor.shape[1], int64_t{7});
        }

        TEST(Unit_TensorFactory, PlacesTheStorageOnTheGivenDeviceAndResidency)
        {
            const TensorFactory factory(TestEnvironment::runtime());
            const std::size_t last = TestEnvironment::device_count() - 1;
            const std::vector<int64_t> dims{4};
            const Tensor tensor = factory.create(span_of(dims), Core::DataType::Float32,
                                                 SushiRuntime::Residency::Device, last);

            EXPECT_EQ(tensor.storage->device_index(), last);
            EXPECT_EQ(tensor.storage->residency(), SushiRuntime::Residency::Device);
        }
    } // namespace Test
} // namespace SushiBLAS
