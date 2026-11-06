/**************************************************************************/
/* test_blend_transforms.cpp                                              */
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

/**
 * @file test_blend_transforms.cpp
 * @brief Tests the SYCL blend of transforms against the reference one.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blend_kernels/reference/blend_transforms.hpp>
#include <SushiBLAS/blend_kernels/sycl/blend_transforms.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the sizes of one blend and the index values its influences carry.
            struct BlendShape
            {
                    int64_t points;       ///< Points to blend.
                    int64_t transforms;   ///< Transforms to choose from.
                    int64_t influences;   ///< Influences per point.
                    int64_t components;   ///< 3 for a position, 4 for a homogeneous point.
                    int64_t rows;         ///< 3 for an implied last row, 4 for a full transform.
                    int64_t first_index;  ///< The smallest index; below zero it is out of range.
                    int64_t index_period; ///< How many consecutive indices repeat.
            };

            /// @brief Holds what the device wrote and what the reference wrote, both in double.
            struct BlendOutcome
            {
                    std::vector<double> device;
                    std::vector<double> reference;
            };

            /// @brief Returns indices that run from first_index for a period, so a blend can meet both ends of an out-of-range one.
            template<typename Index>
            std::vector<Index> indices_around(std::size_t count, const BlendShape& shape)
            {
                std::vector<Index> indices(count);
                for (std::size_t i = 0; i < count; ++i)
                    indices[i] = static_cast<Index>(shape.first_index +
                                                    static_cast<int64_t>(i % static_cast<std::size_t>(shape.index_period)));
                return indices;
            }

            /// @brief Returns the values widened to double.
            template<typename T>
            std::vector<double> widened(const std::vector<T>& values)
            {
                return std::vector<double>(values.begin(), values.end());
            }

            /**
             * @brief Blends small integers on the queue and on the host.
             *
             * Every value is a small integer, so both kernels compute without rounding.
             */
            template<typename T, typename Index>
            BlendOutcome run_blend(sycl::queue& queue, const BlendShape& shape)
            {
                const std::size_t slots = static_cast<std::size_t>(shape.points * shape.influences);
                const std::size_t outputs = static_cast<std::size_t>(shape.points * shape.components);
                const std::vector<T> points = small_integers<T>(outputs, 1);
                const std::vector<T> matrices =
                    small_integers<T>(static_cast<std::size_t>(shape.transforms * shape.rows * 4), 2);
                const std::vector<T> weights = small_integers<T>(slots, 3);
                const std::vector<Index> indices = indices_around<Index>(slots, shape);
                const SharedArray<T> device_points(queue, points);
                const SharedArray<T> device_matrices(queue, matrices);
                const SharedArray<T> device_weights(queue, weights);
                const SharedArray<Index> device_indices(queue, indices);
                SharedArray<T> device_out(queue, outputs);
                std::vector<T> expected(outputs);

                Kernels::Reference::blend_transforms(shape.points, shape.components, shape.rows,
                                                     shape.influences, shape.transforms, points.data(),
                                                     matrices.data(), indices.data(), weights.data(),
                                                     expected.data());
                Kernels::SYCL::blend_transforms(queue, shape.points, shape.components, shape.rows,
                                                shape.influences, shape.transforms, device_points.get(),
                                                device_matrices.get(), device_indices.get(),
                                                device_weights.get(), device_out.get(), {}).wait();

                return {widened(device_out.values()), widened(expected)};
            }

            /// @brief One shape, one scalar type and one index type of a blend.
            struct BlendRow
            {
                    const char* name;
                    BlendShape shape;
                    BlendOutcome (*run)(sycl::queue&, const BlendShape&);
            };

            /// @brief Returns the shape and index-type rows, and the one-point one-influence float row.
            std::vector<BlendRow> blend_rows()
            {
                constexpr int64_t K_POINTS = 37;
                constexpr int64_t K_TRANSFORMS = 4;
                constexpr int64_t K_INFLUENCES = 4;
                constexpr int64_t K_FIRST_INDEX = -1;
                constexpr int64_t K_INDEX_PERIOD = 6;
                const auto shape = [](int64_t components, int64_t rows)
                {
                    return BlendShape{K_POINTS, K_TRANSFORMS, K_INFLUENCES, components, rows,
                                      K_FIRST_INDEX, K_INDEX_PERIOD};
                };
                return {
                    {"Position_ThreeRows_Int32", shape(3, 3), &run_blend<double, int32_t>},
                    {"Position_ThreeRows_Int64", shape(3, 3), &run_blend<double, int64_t>},
                    {"Position_FourRows_Int32", shape(3, 4), &run_blend<double, int32_t>},
                    {"Position_FourRows_Int64", shape(3, 4), &run_blend<double, int64_t>},
                    {"Homogeneous_ThreeRows_Int32", shape(4, 3), &run_blend<double, int32_t>},
                    {"Homogeneous_ThreeRows_Int64", shape(4, 3), &run_blend<double, int64_t>},
                    {"Homogeneous_FourRows_Int32", shape(4, 4), &run_blend<double, int32_t>},
                    {"Homogeneous_FourRows_Int64", shape(4, 4), &run_blend<double, int64_t>},
                    {"OnePointOneInfluenceFloat", BlendShape{1, 1, 1, 3, 3, 0, 1},
                     &run_blend<float, int32_t>},
                };
            }

            /// @brief Names a row, which gtest takes as the test's suffix.
            std::string blend_row_name(const ::testing::TestParamInfo<BlendRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Runs the kernel on the host CPU queue, one row at a time. */
        class Medium_SYCLBlendTransforms : public HostDeviceTest, public ::testing::WithParamInterface<BlendRow>
        {

        };

        TEST_P(Medium_SYCLBlendTransforms, MatchesTheReferenceInEveryShapeAndIndexType)
        {
            const BlendRow& row = GetParam();

            const BlendOutcome outcome = row.run(queue(), row.shape);

            EXPECT_EQ(outcome.device, outcome.reference);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_SYCLBlendTransforms, ::testing::ValuesIn(blend_rows()),
                                 blend_row_name);
    } // namespace Test
} // namespace SushiBLAS
