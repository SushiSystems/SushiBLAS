/**************************************************************************/
/* test_runtime_shapes.cpp                                                */
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
 * @file test_runtime_shapes.cpp
 * @brief Tests create_tensor and reshape with dimensions held in a std::vector.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds a rank-3 shape known only at run time. */
        const std::vector<int64_t> K_DIMS{2, 3, 4};

        /** @brief Holds 2 * 3 * 4, the element count of K_DIMS. */
        constexpr int64_t K_ELEMENTS = 24;

        /** @brief Holds the row-major strides of K_DIMS: {3 * 4, 4, 1}. */
        const std::vector<int64_t> K_ROW_MAJOR_STRIDES{12, 4, 1};

        namespace
        {
            /** @brief Names a create_tensor call by the dtype it passes, or by passing none. */
            struct CreateRow
            {
                const char* name;                  /**< The row name a failure shows. */
                std::optional<Core::DataType> arg; /**< The dtype argument; empty for the default call. */
                Core::DataType expected;           /**< The dtype the tensor must have. */
            };

            /** @brief Returns the default call and an explicit Float64 call. */
            std::vector<CreateRow> create_rows()
            {
                return {{"DefaultDtype", std::nullopt, Core::DataType::Float32},
                        {"ExplicitFloat64", Core::DataType::Float64, Core::DataType::Float64}};
            }
        } // namespace

        /** @brief Runs one row of create_rows() on a host engine. */
        class Small_RuntimeShapesCreate : public ::testing::WithParamInterface<CreateRow>,
                                          public HostEngineTest
        {

        };

        TEST_P(Small_RuntimeShapesCreate, TensorTakesAVectorShapeAndGivesItsRowMajorStrides)
        {
            const CreateRow row = GetParam();

            const Tensor t = row.arg ? engine->create_tensor(K_DIMS, *row.arg)
                                     : engine->create_tensor(K_DIMS);

            ASSERT_EQ(t.rank, 3);
            EXPECT_EQ(t.num_elements, K_ELEMENTS);
            EXPECT_EQ(t.dtype, row.expected);
            EXPECT_EQ(std::vector<int64_t>(t.shape.begin(), t.shape.begin() + t.rank), K_DIMS);
            EXPECT_EQ(std::vector<int64_t>(t.strides.begin(), t.strides.begin() + t.rank), K_ROW_MAJOR_STRIDES);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_RuntimeShapesCreate, ::testing::ValuesIn(create_rows()),
            [](const ::testing::TestParamInfo<CreateRow>& info)
            { return std::string(info.param.name); });

        /** @brief Tests that the allocation size of a negative dimension is refused before any allocation. */
        TEST(Small_RuntimeShapes, AllocationRefusesANegativeDimension)
        {
            const std::vector<int64_t> negative{2, -3};

            EXPECT_THROW((void)TensorFactory::allocation_bytes(
                             SushiRuntime::span<const int64_t>(negative.data(), negative.size()),
                             Core::DataType::Float32),
                         std::runtime_error);
        }
    } // namespace Test
} // namespace SushiBLAS
