/**************************************************************************/
/* test_pool_avg.cpp                                                      */
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
 * @file test_pool_avg.cpp
 * @brief Tests the reference average pool against hand values and its backward as its adjoint.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <functional>
#include <cstdint>
#include <numeric>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/reference/pool_avg.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include "support/tolerance.hpp"
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns the sum of |a[i] * b[i]|, the magnitude a dot product's rounding bound scales with. */
            double sum_of_absolute_products(const std::vector<double>& a, const std::vector<double>& b)
            {
                return std::inner_product(a.begin(), a.end(), b.begin(), 0.0, std::plus<double>(),
                                          [](double left, double right) { return std::abs(left * right); });
            }
        } // namespace

        TEST(Small_ReferencePoolAvg, AveragesEachWindowOverItsFullArea)
        {
            Spatial::ConvGeometry geometry;
            geometry.kernel_h = 2;
            geometry.kernel_w = 2;
            geometry.stride_h = 2;
            geometry.stride_w = 2;
            const Spatial::WindowSpan span = nhwc_span(1, 2, 4, 1, geometry, 0);
            const std::vector<float> in{1, 2, 3, 4, 5, 6, 7, 8};
            std::vector<float> out(2, 0.0f);

            Kernels::Reference::pool_avg(span, in.data(), out.data());

            EXPECT_EQ(out, (std::vector<float>{3.5f, 5.5f}));
        }

        TEST(Small_ReferencePoolAvg, TheBackwardIsTheAdjoint)
        {
            // A window area of 6 makes each average a sixth, so the dot products agree within a sum-of-products bound.
            const Spatial::WindowSpan span = nhwc_span(2, 7, 6, 2, uneven_geometry(), 1);
            const std::vector<double> x = small_integers<double>(span.input_elements(), 1);
            const std::vector<double> dy = small_integers<double>(span.output_elements(), 2);
            std::vector<double> y(static_cast<std::size_t>(span.output_elements()));
            std::vector<double> dx(static_cast<std::size_t>(span.input_elements()));

            Kernels::Reference::pool_avg(span, x.data(), y.data());
            Kernels::Reference::pool_avg_backward(span, dy.data(), dx.data());

            const double lhs = std::inner_product(y.begin(), y.end(), dy.begin(), 0.0);
            const double rhs = std::inner_product(x.begin(), x.end(), dx.begin(), 0.0);
            const double magnitude = sum_of_absolute_products(y, dy) + sum_of_absolute_products(x, dx);
            EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_sum_error<double>(
                lhs, rhs, magnitude, y.size() + x.size())));
        }
    } // namespace Test
} // namespace SushiBLAS
