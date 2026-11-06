/**************************************************************************/
/* test_pool_max.cpp                                                      */
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
 * @file test_pool_max.cpp
 * @brief Tests the reference max pool, the index it records and the backward that uses it.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/reference/pool_max.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Holds what one max pool of the 1 x 2 x 4 sample wrote. */
            struct PooledSample
            {
                    std::vector<float> values;     /**< The window maxima. */
                    std::vector<int64_t> indices;  /**< The flat input positions of the maxima. */
            };

            /** @brief Pools the 1 x 2 x 4 sample 1 9 3 4 / 5 6 7 8 with 2 x 2 windows at stride 2. */
            PooledSample pool_the_sample()
            {
                Spatial::ConvGeometry geometry;
                geometry.kernel_h = 2;
                geometry.kernel_w = 2;
                geometry.stride_h = 2;
                geometry.stride_w = 2;
                const Spatial::WindowSpan span = nhwc_span(1, 2, 4, 1, geometry, 0);
                const std::vector<float> in{1, 9, 3, 4, 5, 6, 7, 8};
                PooledSample out{std::vector<float>(2, 0.0f), std::vector<int64_t>(2, -1)};

                Kernels::Reference::pool_max(span, in.data(), out.values.data(), out.indices.data());
                return out;
            }
        } // namespace

        TEST(Small_ReferencePoolMax, KeepsTheLargestTapOfEachWindow)
        {
            EXPECT_EQ(pool_the_sample().values, (std::vector<float>{9.0f, 8.0f}));
        }

        TEST(Small_ReferencePoolMax, RecordsTheFlatPositionOfTheLargestTap)
        {
            EXPECT_EQ(pool_the_sample().indices, (std::vector<int64_t>{1, 7}));
        }

        TEST(Small_ReferencePoolMax, TheBackwardSendsEachGradientToItsArgmax)
        {
            Spatial::ConvGeometry geometry;
            geometry.kernel_h = 2;
            geometry.kernel_w = 2;
            geometry.stride_h = 2;
            geometry.stride_w = 2;
            const Spatial::WindowSpan span = nhwc_span(1, 2, 4, 1, geometry, 0);
            const std::vector<float> dy{10.0f, 20.0f};
            const std::vector<int64_t> indices{1, 7};
            std::vector<float> dx(8, -1.0f);

            Kernels::Reference::pool_max_backward(span, dy.data(), indices.data(), dx.data());

            EXPECT_EQ(dx, (std::vector<float>{0, 10, 0, 0, 0, 0, 0, 20}));
        }
    } // namespace Test
} // namespace SushiBLAS
