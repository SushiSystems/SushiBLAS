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
 * @brief Tests TensorFactory and Engine::create_tensor: shapes, layout, device index and size.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/core/common.hpp>
#include "test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Builds one factory over the test environment's context for each test. */
        class Unit_TensorFactoryTest : public SushiBLASTest
        {
            protected:
                /** @brief Returns the context the factory allocates from. */
                [[nodiscard]] static SushiRuntime::Execution::RuntimeContext& context()
                {
                    return TestEnvironment::get_context();
                }

                /** @brief Returns a view of dims, which must outlive the call it is passed to. */
                [[nodiscard]] static SushiRuntime::span<const int64_t> span_of(
                    const std::vector<int64_t>& dims)
                {
                    return SushiRuntime::span<const int64_t>(dims.data(), dims.size());
                }

                /// @brief The factory under test, row-major.
                TensorFactory factory_{context()};
        };

        /** @brief Tests that a shape with a zero dimension keeps its rank and allocates nothing. */
        TEST_F(Unit_TensorFactoryTest, AZeroDimensionGivesATensorWithoutStorage)
        {
            const std::vector<int64_t> dims = {0, 3};
            const std::size_t live_before = context().live_usm_count();

            const Tensor tensor = factory_.create(span_of(dims));

            EXPECT_TRUE(tensor.storage == nullptr);
            EXPECT_EQ(0, tensor.num_elements);
            EXPECT_EQ(2, tensor.rank);
            EXPECT_EQ(0, tensor.shape[0]);
            EXPECT_EQ(3, tensor.shape[1]);
            EXPECT_EQ(live_before, context().live_usm_count());
        }

        /** @brief Tests that the storage reports the device index the factory was asked for. */
        TEST_F(Unit_TensorFactoryTest, TheDeviceIndexReachesTheStorage)
        {
            const std::vector<int64_t> dims = {4};

            for (std::size_t device = 0; device < context().device_count(); ++device)
            {
                SCOPED_TRACE(device);
                const Tensor tensor = factory_.create(span_of(dims), Core::DataType::Float32,
                                                      SushiRuntime::API::Residency::Shared, device);

                ASSERT_TRUE(tensor.storage != nullptr);
                EXPECT_EQ(device, tensor.storage->get_device_index());
            }
        }

        /** @brief Tests that a device index past the last queue throws, for an empty shape too. */
        TEST_F(Unit_TensorFactoryTest, ADeviceIndexOutOfRangeThrows)
        {
            const std::vector<int64_t> dims = {4};
            const std::vector<int64_t> empty_dims = {0, 3};
            const std::size_t past_the_last = context().device_count();

            EXPECT_THROW((void)factory_.create(span_of(dims), Core::DataType::Float32,
                                               SushiRuntime::API::Residency::Shared, past_the_last),
                         std::runtime_error);
            EXPECT_THROW((void)factory_.create(span_of(empty_dims), Core::DataType::Float32,
                                               SushiRuntime::API::Residency::Shared, past_the_last),
                         std::runtime_error);
        }

        /** @brief Tests that a brace list, a vector and a span give equal shapes and strides. */
        TEST_F(Unit_TensorFactoryTest, BraceListVectorAndSpanAgree)
        {
            const std::vector<int64_t> dims = {2, 3, 4};

            const Tensor from_braces = engine->create_tensor({2, 3, 4});
            const Tensor from_vector = engine->create_tensor(dims);
            const Tensor from_span = engine->create_tensor(span_of(dims));

            EXPECT_EQ(3, from_braces.rank);
            EXPECT_EQ(from_braces.rank, from_vector.rank);
            EXPECT_EQ(from_braces.rank, from_span.rank);
            EXPECT_EQ(from_braces.shape, from_vector.shape);
            EXPECT_EQ(from_braces.shape, from_span.shape);
            EXPECT_EQ(from_braces.strides, from_vector.strides);
            EXPECT_EQ(from_braces.strides, from_span.strides);
        }

        /** @brief Tests that a column-major engine creates column-major tensors. */
        TEST_F(Unit_TensorFactoryTest, EngineCreateTensorUsesTheEngineLayout)
        {
            reinit_engine(Core::Layout::ColumnMajor);

            const Tensor tensor = engine->create_tensor({2, 3});

            EXPECT_EQ(Core::Layout::ColumnMajor, engine->get_tensor_factory().get_layout());
            EXPECT_EQ(Core::Layout::ColumnMajor, tensor.layout);
            EXPECT_EQ(1, tensor.strides[0]);
            EXPECT_EQ(2, tensor.strides[1]);
        }

        /** @brief Tests that twelve requested bytes are stored as whole alignment units. */
        TEST_F(Unit_TensorFactoryTest, StorageSizeIsRoundedUpToTheAlignment)
        {
            const Tensor tensor = engine->create_tensor({3}, Core::DataType::Float32);

            ASSERT_TRUE(tensor.storage != nullptr);
            const std::size_t stored_bytes = tensor.storage->size_bytes();
            EXPECT_EQ(0u, stored_bytes % SushiRuntime::Core::DEFAULT_ALIGNMENT);
            EXPECT_GE(stored_bytes, std::size_t{12});
        }
    } // namespace Test
} // namespace SushiBLAS
