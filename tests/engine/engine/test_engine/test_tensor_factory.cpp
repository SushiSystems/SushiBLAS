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
 * @brief Tests TensorFactory and Engine::create_tensor: shapes, layout, device index and size.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns a view of dims, which must outlive the call it is passed to. */
            SushiRuntime::span<const int64_t> span_of(const std::vector<int64_t>& dims)
            {
                return SushiRuntime::span<const int64_t>(dims.data(), dims.size());
            }

            /** @brief Expects the shape {2, 3, 4} with the row-major strides {12, 4, 1}. */
            void expect_two_by_three_by_four(const Tensor& tensor)
            {
                EXPECT_EQ(tensor.rank, 3);
                EXPECT_EQ(tensor.shape[0], 2);
                EXPECT_EQ(tensor.shape[1], 3);
                EXPECT_EQ(tensor.shape[2], 4);
                EXPECT_EQ(tensor.strides[0], 12);
                EXPECT_EQ(tensor.strides[1], 4);
                EXPECT_EQ(tensor.strides[2], 1);
            }
        } // namespace

        /** @brief Gives each test a fresh host engine. */
        class Small_TensorFactoryTest : public HostEngineTest
        {

        };

        /** @brief Tests that a device index past the last queue throws, on a host factory's one queue. */
        TEST(Small_TensorFactory, ADeviceIndexOutOfRangeThrows)
        {
            const TensorFactory factory{TensorFactory::Host{}};
            const std::vector<int64_t> dims = {4};

            EXPECT_THROW((void)factory.create(span_of(dims), Core::DataType::Float32,
                                              SushiRuntime::Residency::Shared, 1),
                         std::runtime_error);
        }

        /** @brief Tests that the device-index refusal comes before the empty-shape shortcut. */
        TEST(Small_TensorFactory, ADeviceIndexOutOfRangeThrowsForAnEmptyShapeToo)
        {
            const TensorFactory factory{TensorFactory::Host{}};
            const std::vector<int64_t> empty_dims = {0, 3};

            EXPECT_THROW((void)factory.create(span_of(empty_dims), Core::DataType::Float32,
                                              SushiRuntime::Residency::Shared, 1),
                         std::runtime_error);
        }

        /** @brief Tests that a brace list gives the shape {2, 3, 4} and its row-major strides. */
        TEST_F(Small_TensorFactoryTest, BraceListGivesTheShapeAndStrides)
        {
            expect_two_by_three_by_four(engine->create_tensor({2, 3, 4}));
        }

        /** @brief Tests that a vector gives the shape {2, 3, 4} and its row-major strides. */
        TEST_F(Small_TensorFactoryTest, VectorGivesTheShapeAndStrides)
        {
            const std::vector<int64_t> dims = {2, 3, 4};

            expect_two_by_three_by_four(engine->create_tensor(dims));
        }

        /** @brief Tests that a span gives the shape {2, 3, 4} and its row-major strides. */
        TEST_F(Small_TensorFactoryTest, SpanGivesTheShapeAndStrides)
        {
            const std::vector<int64_t> dims = {2, 3, 4};

            expect_two_by_three_by_four(engine->create_tensor(span_of(dims)));
        }
    } // namespace Test
} // namespace SushiBLAS
